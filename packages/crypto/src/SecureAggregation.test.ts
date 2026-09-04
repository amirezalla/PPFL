import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  SecureAggregationClient,
  SecureAggregationCoordinator,
  ShamirSecretSharing,
} from './SecureAggregation.js';

test('ShamirSecretSharing: reconstructing from exactly the threshold recovers the secret', () => {
  const sss = new ShamirSecretSharing();
  const secret = 123456789n;
  const shares = sss.split(secret, 5, 3);
  const recovered = sss.reconstruct(shares.slice(0, 3));
  assert.equal(recovered, secret);
});

test('ShamirSecretSharing: any 3-of-5 subset reconstructs the same secret', () => {
  const sss = new ShamirSecretSharing();
  const secret = 42n;
  const shares = sss.split(secret, 5, 3);
  assert.equal(sss.reconstruct([shares[0]!, shares[2]!, shares[4]!]), secret);
  assert.equal(sss.reconstruct([shares[1]!, shares[3]!, shares[4]!]), secret);
});

test('ShamirSecretSharing: fewer than the threshold does not reconstruct the secret', () => {
  const sss = new ShamirSecretSharing();
  const secret = 999999n;
  const shares = sss.split(secret, 5, 3);
  const wrong = sss.reconstruct(shares.slice(0, 2));
  assert.notEqual(wrong, secret);
});

test('ShamirSecretSharing: split rejects an invalid threshold', () => {
  const sss = new ShamirSecretSharing();
  assert.throws(() => sss.split(1n, 5, 1), RangeError);
  assert.throws(() => sss.split(1n, 5, 6), RangeError);
});

test('SecureAggregationClient.shareSeed rejects a 1-of-1 threshold (no real secret sharing)', () => {
  const client = new SecureAggregationClient('solo');
  assert.throws(() => client.shareSeed(1, 1), RangeError);
});

test('SecureAggregationCoordinator recovers the exact true sum when one client drops out', () => {
  // 4 clients set up pairwise masks with every other client, but "dave"
  // drops out and never submits - his surviving peers (alice, bob, carol)
  // are each left carrying one uncancelled pairwise term keyed to dave.
  const clientIds = ['alice', 'bob', 'carol', 'dave'];
  const clients = clientIds.map((id) => new SecureAggregationClient(id));
  const peerHandles = clients.map((c) => ({ clientId: c.clientId, publicKey: c.publicKey }));
  const [alice, bob, carol, dave] = clients;

  const trueVectors = new Map<string, Float32Array>([
    ['alice', new Float32Array([10, 20])],
    ['bob', new Float32Array([30, 40])],
    ['carol', new Float32Array([50, 60])],
  ]);

  // Only the three survivors submit; each still masks against ALL four
  // original peers, since masking happens before anyone knows who will drop.
  const survivors = [alice!, bob!, carol!];
  const masked = survivors.map((c) => {
    const peers = peerHandles.filter((p) => p.clientId !== c.clientId);
    return c.maskVector(trueVectors.get(c.clientId)!, peers);
  });

  const sumVectors = (vectors: Float32Array[]): Float32Array => {
    const out = new Float32Array(vectors[0]!.length);
    for (const v of vectors) for (let i = 0; i < v.length; i++) out[i] = (out[i] ?? 0) + (v[i] ?? 0);
    return out;
  };

  const maskedSum = sumVectors(masked);
  const trueSum = sumVectors([...trueVectors.values()]);

  // Step 1: strip the three survivors' self-masks (dave's is moot - he
  // never contributed a self-mask to the sum in the first place).
  const coordinator = new SecureAggregationCoordinator();
  const survivorSelfMaskShares = survivors.map((c) => c.shareSeed(3, 2));
  const afterSelfMaskRemoval = coordinator.removeSelfMasks(maskedSum, survivorSelfMaskShares);

  // Sanity check: pairwise terms against dave are still stuck in there.
  assert.notEqual(afterSelfMaskRemoval[0], trueSum[0]);

  // Step 2: dave's private key is reconstructed from shares his peers
  // (including survivors) were holding, letting the coordinator recompute
  // and cancel every pairwise term dave's absence left behind.
  const davePrivateKeyShares = dave!.sharePrivateKey(3, 2);
  const correction = coordinator.recoverDroppedClientContribution(
    'dave',
    davePrivateKeyShares,
    survivors.map((c) => ({ clientId: c.clientId, publicKey: c.publicKey })),
    trueVectors.get('alice')!.length,
  );
  const recovered = coordinator.applyDroppedClientCorrections(afterSelfMaskRemoval, [correction]);

  for (let i = 0; i < trueSum.length; i++) {
    assert.ok(Math.abs((recovered[i] ?? 0) - (trueSum[i] ?? 0)) < 1e-3, `index ${i} did not match`);
  }
});

test('SecureAggregationCoordinator.reconstructPrivateKey rejects an unknown curve', () => {
  const client = new SecureAggregationClient('solo');
  const coordinator = new SecureAggregationCoordinator();
  const shares = client.sharePrivateKey(3, 2);
  assert.throws(() => coordinator.reconstructPrivateKey(shares, 'not-a-real-curve'), RangeError);
});

test('SecureAggregationClient: pairwise masks cancel across a cohort once self-masks are removed', () => {
  const clientIds = ['alice', 'bob', 'carol'];
  const clients = clientIds.map((id) => new SecureAggregationClient(id));
  const peerHandles = clients.map((c) => ({ clientId: c.clientId, publicKey: c.publicKey }));

  const trueVectors = clientIds.map((_, i) => new Float32Array([10 * (i + 1), 20 * (i + 1), 30 * (i + 1)]));
  const masked = clients.map((c, i) => {
    const peers = peerHandles.filter((p) => p.clientId !== c.clientId);
    return c.maskVector(trueVectors[i]!, peers);
  });

  const sumVectors = (vectors: Float32Array[]): Float32Array => {
    const out = new Float32Array(vectors[0]!.length);
    for (const v of vectors) for (let i = 0; i < v.length; i++) out[i] = (out[i] ?? 0) + (v[i] ?? 0);
    return out;
  };

  const maskedSum = sumVectors(masked);
  const trueSum = sumVectors(trueVectors);

  // Sanity check: without self-mask removal, the masked sum is NOT the true
  // sum - pairwise masks alone cancel, but self-mask noise remains.
  assert.notEqual(maskedSum[0], trueSum[0]);

  // Every client splits its self-seed 2-of-2 for its peers; the coordinator
  // reconstructs and strips every self-mask to recover the exact true sum.
  const sharesPerClient = clients.map((c) => c.shareSeed(2, 2));
  const coordinator = new SecureAggregationCoordinator();
  const recovered = coordinator.removeSelfMasks(maskedSum, sharesPerClient);

  for (let i = 0; i < trueSum.length; i++) {
    assert.ok(Math.abs((recovered[i] ?? 0) - (trueSum[i] ?? 0)) < 1e-3, `index ${i} did not match`);
  }
});
