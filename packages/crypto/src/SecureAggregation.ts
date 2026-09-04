import { createECDH, createHash, randomBytes, type ECDH } from 'node:crypto';

/**
 * Secure aggregation (Bonawitz et al.-style): every pair of clients derives a
 * shared pairwise mask via ECDH and adds it to their local update with an
 * opposite sign convention (lower clientId = +1, higher = -1). When the
 * aggregator sums every client's masked vector, all pairwise masks cancel
 * out algebraically, revealing only the true sum/average - the aggregator
 * never observes an individual client's raw weights.
 *
 * Each client also blends in a private self-mask derived from a seed that is
 * split via Shamir's Secret Sharing across the cohort. If a client drops out
 * mid-round, a threshold of surviving peers can reconstruct that seed and the
 * aggregator subtracts the corresponding mask out of the final sum, so a
 * straggler's mask never permanently corrupts the round.
 *
 * Dropout resilience for the PAIRWISE masks works differently: a dropped
 * client's own submission never arrives, so its surviving peers are each
 * left carrying an uncancelled pairwise term keyed to that dropped client.
 * Rather than have every surviving peer separately reveal its pairwise
 * seed, each client instead Shamir-shares its ECDH PRIVATE KEY
 * (`sharePrivateKey`). Reconstructing a dropped client's private key lets
 * the coordinator recompute every pairwise seed that client used - against
 * each surviving peer's already-public public key - unilaterally, and
 * cancel all of those stuck terms in one pass
 * (`SecureAggregationCoordinator.recoverDroppedClientContribution`).
 */

const DEFAULT_PRIME = (1n << 127n) - 1n; // Mersenne prime 2^127 - 1
const KEY_SHARING_PRIME = (1n << 521n) - 1n; // Mersenne prime 2^521 - 1: comfortably larger than any EC private key scalar used here

const CURVE_KEY_BYTE_LENGTHS: Record<string, number> = {
  prime256v1: 32,
  secp384r1: 48,
  secp521r1: 66,
};

export interface Share {
  x: bigint;
  y: bigint;
}

function mod(a: bigint, p: bigint): bigint {
  const r = a % p;
  return r < 0n ? r + p : r;
}

function modInverse(a: bigint, p: bigint): bigint {
  let [oldR, r] = [mod(a, p), p];
  let [oldS, s] = [1n, 0n];
  while (r !== 0n) {
    const q = oldR / r;
    [oldR, r] = [r, oldR - q * r];
    [oldS, s] = [s, oldS - q * s];
  }
  return mod(oldS, p);
}

function randomFieldElement(prime: bigint): bigint {
  const byteLength = Math.ceil(prime.toString(2).length / 8);
  let value: bigint;
  do {
    value = BigInt('0x' + randomBytes(byteLength).toString('hex'));
  } while (value >= prime);
  return value;
}

/** Shamir's Secret Sharing over GF(prime), used to split mask seeds for dropout recovery. */
export class ShamirSecretSharing {
  constructor(private readonly prime: bigint = DEFAULT_PRIME) {}

  split(secret: bigint, totalShares: number, threshold: number): Share[] {
    if (threshold < 2 || threshold > totalShares) {
      throw new RangeError('threshold must be between 2 and totalShares');
    }
    if (secret < 0n || secret >= this.prime) {
      throw new RangeError('secret must lie within the field');
    }

    const coefficients = [secret];
    for (let i = 1; i < threshold; i++) {
      coefficients.push(randomFieldElement(this.prime));
    }

    const shares: Share[] = [];
    for (let x = 1; x <= totalShares; x++) {
      let y = 0n;
      let xPow = 1n;
      for (const c of coefficients) {
        y = mod(y + c * xPow, this.prime);
        xPow = mod(xPow * BigInt(x), this.prime);
      }
      shares.push({ x: BigInt(x), y });
    }
    return shares;
  }

  /** Lagrange interpolation at x=0 to recover the secret from >= threshold shares. */
  reconstruct(shares: Share[]): bigint {
    let secret = 0n;
    for (let i = 0; i < shares.length; i++) {
      const { x: xi, y: yi } = shares[i]!;
      let numerator = 1n;
      let denominator = 1n;
      for (let j = 0; j < shares.length; j++) {
        if (i === j) continue;
        const xj = shares[j]!.x;
        numerator = mod(numerator * -xj, this.prime);
        denominator = mod(denominator * (xi - xj), this.prime);
      }
      const lagrangeCoefficient = mod(numerator * modInverse(denominator, this.prime), this.prime);
      secret = mod(secret + yi * lagrangeCoefficient, this.prime);
    }
    return secret;
  }
}

function deriveSeed32(seed: bigint): number {
  const hash = createHash('sha256').update(seed.toString(16)).digest();
  return hash.readUInt32BE(0);
}

/** mulberry32: fast, deterministic PRNG used to expand a shared seed into a mask vector. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function expandMaskVector(seed: bigint, length: number): Float32Array {
  const rng = mulberry32(deriveSeed32(seed));
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = rng() * 2 - 1; // uniform in [-1, 1)
  }
  return out;
}

export function generateKeyPair(curve = 'prime256v1'): { privateKey: ECDH; publicKey: Buffer } {
  const ecdh = createECDH(curve);
  ecdh.generateKeys();
  return { privateKey: ecdh, publicKey: ecdh.getPublicKey() };
}

export function deriveSharedSeed(privateKey: ECDH, peerPublicKey: Buffer): bigint {
  const secret = privateKey.computeSecret(peerPublicKey);
  return BigInt('0x' + secret.toString('hex'));
}

export interface PeerHandle {
  clientId: string;
  publicKey: Buffer;
}

export class SecureAggregationClient {
  private readonly sss = new ShamirSecretSharing();
  private readonly keySss = new ShamirSecretSharing(KEY_SHARING_PRIME);
  private readonly selfSeed: bigint;
  private readonly keyPair: { privateKey: ECDH; publicKey: Buffer };

  constructor(
    public readonly clientId: string,
    keyPair: { privateKey: ECDH; publicKey: Buffer } = generateKeyPair(),
  ) {
    this.keyPair = keyPair;
    this.selfSeed = randomFieldElement(DEFAULT_PRIME);
  }

  get publicKey(): Buffer {
    return this.keyPair.publicKey;
  }

  /**
   * Splits this client's private mask seed into shares distributed to peers
   * at round setup. The aggregator must later obtain >= threshold of these
   * shares (from surviving peers, never from this client directly) to
   * reconstruct the seed and strip this client's self-mask out of the sum -
   * that indirection is what stops the aggregator from ever unmasking a
   * single client's contribution unilaterally.
   */
  shareSeed(totalPeers: number, threshold: number): Share[] {
    return this.sss.split(this.selfSeed, totalPeers, threshold);
  }

  /**
   * Splits this client's ECDH PRIVATE KEY - distinct from `shareSeed`'s
   * self-mask seed - so a threshold of peers can reconstruct it if this
   * client drops out mid-round. Reconstructing the private key lets the
   * coordinator independently recompute the exact pairwise seed this
   * client shared with every surviving peer (ECDH only needs one side's
   * private key plus the other's already-public public key), and so
   * cancel this client's now-permanently-missing half of every pairwise
   * mask it set up. See `SecureAggregationCoordinator.recoverDroppedClientContribution`.
   */
  sharePrivateKey(totalPeers: number, threshold: number): Share[] {
    const scalar = BigInt('0x' + this.keyPair.privateKey.getPrivateKey().toString('hex'));
    return this.keySss.split(scalar, totalPeers, threshold);
  }

  /**
   * Masks a local weight-delta vector before it ever leaves the device:
   * adds a signed pairwise mask per peer (algebraically cancels once the
   * aggregator sums every pair of contributions) plus a private self-mask.
   * Unlike the pairwise term, the self-mask does NOT cancel on its own - the
   * aggregator must reconstruct it from peer shares (see `shareSeed`) and
   * remove it via `SecureAggregationCoordinator.removeSelfMasks` before the
   * sum reflects the true plaintext total.
   */
  maskVector(vector: Float32Array, peers: PeerHandle[]): Float32Array {
    const masked = new Float32Array(vector);

    for (const peer of peers) {
      const sharedSeed = deriveSharedSeed(this.keyPair.privateKey, peer.publicKey);
      const mask = expandMaskVector(sharedSeed, vector.length);
      const sign = this.clientId < peer.clientId ? 1 : -1;
      for (let i = 0; i < masked.length; i++) {
        masked[i] = (masked[i] ?? 0) + sign * (mask[i] ?? 0);
      }
    }

    const selfMask = expandMaskVector(this.selfSeed, vector.length);
    for (let i = 0; i < masked.length; i++) {
      masked[i] = (masked[i] ?? 0) + (selfMask[i] ?? 0);
    }
    return masked;
  }

  /** Subtracts the mask expanded from a single reconstructed seed out of a running sum. */
  static removeMaskForSeed(aggregate: Float32Array, recoveredSeed: bigint): Float32Array {
    const mask = expandMaskVector(recoveredSeed, aggregate.length);
    const out = new Float32Array(aggregate.length);
    for (let i = 0; i < aggregate.length; i++) {
      out[i] = (aggregate[i] ?? 0) - (mask[i] ?? 0);
    }
    return out;
  }
}

/**
 * Aggregator-side counterpart to `SecureAggregationClient`. Once every
 * (surviving) client's masked vector has been summed, pairwise masks
 * between any two CONTRIBUTING clients have already cancelled out; what
 * remains is the true sum, plus every contributor's uncancelled self-mask,
 * plus one uncancelled pairwise term for every surviving peer paired with
 * a client that dropped out. `removeSelfMasks` handles the former;
 * `recoverDroppedClientContribution` handles the latter.
 */
export class SecureAggregationCoordinator {
  private readonly sss = new ShamirSecretSharing();
  private readonly keySss = new ShamirSecretSharing(KEY_SHARING_PRIME);

  reconstructSeed(shares: Share[]): bigint {
    return this.sss.reconstruct(shares);
  }

  /** Strips every contributing client's self-mask out of the masked sum, in place of a fresh copy. */
  removeSelfMasks(maskedSum: Float32Array, contributingClientShares: Share[][]): Float32Array {
    let result = maskedSum;
    for (const shares of contributingClientShares) {
      const seed = this.reconstructSeed(shares);
      result = SecureAggregationClient.removeMaskForSeed(result, seed);
    }
    return result;
  }

  /** Reconstructs a dropped client's ECDH private key from >= threshold shares gathered from surviving peers. */
  reconstructPrivateKey(shares: Share[], curve = 'prime256v1'): ECDH {
    const byteLength = CURVE_KEY_BYTE_LENGTHS[curve];
    if (!byteLength) throw new RangeError(`Unknown curve "${curve}"; add its key byte length to CURVE_KEY_BYTE_LENGTHS.`);

    const scalar = this.keySss.reconstruct(shares);
    let hex = scalar.toString(16);
    if (hex.length % 2 !== 0) hex = `0${hex}`;
    const raw = Buffer.from(hex, 'hex');
    if (raw.length > byteLength) {
      throw new RangeError(`Reconstructed private key is longer than curve "${curve}" allows.`);
    }
    const padded = raw.length === byteLength ? raw : Buffer.concat([Buffer.alloc(byteLength - raw.length), raw]);

    const ecdh = createECDH(curve);
    ecdh.setPrivateKey(padded);
    return ecdh;
  }

  /**
   * Once a client is confirmed dropped (it never submitted this round),
   * this recomputes - using only the reconstructed private key and every
   * surviving peer's already-public public key, no further peer
   * interaction required - the exact pairwise mask contribution each
   * surviving peer is still carrying for its pairing with the dropped
   * client, and sums them into one correction vector. Adding that
   * correction to the masked sum cancels those otherwise permanently-stuck
   * pairwise terms.
   */
  recoverDroppedClientContribution(
    droppedClientId: string,
    droppedClientPrivateKeyShares: Share[],
    survivingPeers: PeerHandle[],
    vectorLength: number,
    curve = 'prime256v1',
  ): Float32Array {
    const droppedPrivateKey = this.reconstructPrivateKey(droppedClientPrivateKeyShares, curve);
    const correction = new Float32Array(vectorLength);

    for (const peer of survivingPeers) {
      const sharedSeed = deriveSharedSeed(droppedPrivateKey, peer.publicKey);
      const mask = expandMaskVector(sharedSeed, vectorLength);
      // Mirrors maskVector()'s sign convention exactly: this is the sign
      // the DROPPED client would have used for its own contribution to
      // this pair, which is precisely what's missing from the sum.
      const sign = droppedClientId < peer.clientId ? 1 : -1;
      for (let i = 0; i < vectorLength; i++) {
        correction[i] = (correction[i] ?? 0) + sign * (mask[i] ?? 0);
      }
    }

    return correction;
  }

  /** Adds one or more dropped-client correction vectors (from `recoverDroppedClientContribution`) onto the sum. */
  applyDroppedClientCorrections(maskedSum: Float32Array, corrections: Float32Array[]): Float32Array {
    const result = new Float32Array(maskedSum);
    for (const correction of corrections) {
      for (let i = 0; i < result.length; i++) {
        result[i] = (result[i] ?? 0) + (correction[i] ?? 0);
      }
    }
    return result;
  }
}
