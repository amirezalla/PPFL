import { randomBytes } from 'node:crypto';

/**
 * (epsilon, delta)-differential privacy configuration for a single gradient
 * release. `sensitivity` is the L2 sensitivity of the query; when `clipNorm`
 * is supplied the vector is first clipped to that norm and the sensitivity
 * of the clipped query is derived as 2 * clipNorm (add/remove-one-record
 * bound for a single client's contribution).
 */
export interface DifferentialPrivacyConfig {
  epsilon: number;
  delta: number;
  sensitivity: number;
  clipNorm?: number;
}

export interface NoiseResult {
  noised: Float32Array;
  sigma: number;
}

/** Uniform double in [0, 1) drawn from a CSPRNG, with full 53-bit mantissa precision. */
function secureUniform(): number {
  const buf = randomBytes(8);
  const bits = buf.readBigUInt64BE(0) >> 11n;
  return Number(bits) / 2 ** 53;
}

/** Box-Muller transform over CSPRNG uniforms, producing a pair of standard normal draws. */
function secureGaussianPair(): [number, number] {
  let u1 = secureUniform();
  while (u1 <= Number.EPSILON) u1 = secureUniform();
  const u2 = secureUniform();
  const r = Math.sqrt(-2 * Math.log(u1));
  const theta = 2 * Math.PI * u2;
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

export class PrivacyEngine {
  static l2Norm(vec: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < vec.length; i++) {
      const v = vec[i] ?? 0;
      sum += v * v;
    }
    return Math.sqrt(sum);
  }

  /** Rescales the vector so its L2 norm does not exceed maxNorm, bounding per-client sensitivity. */
  static clipByNorm(vec: Float32Array, maxNorm: number): Float32Array {
    const norm = this.l2Norm(vec);
    if (norm <= maxNorm || norm === 0) return vec;
    const scale = maxNorm / norm;
    const out = new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i++) out[i] = (vec[i] ?? 0) * scale;
    return out;
  }

  /** Analytic Gaussian mechanism calibration (Dwork & Roth): sigma for (eps, delta)-DP. */
  static gaussianSigma(epsilon: number, delta: number, sensitivity: number): number {
    if (epsilon <= 0) throw new RangeError('epsilon must be > 0');
    if (delta <= 0 || delta >= 1) throw new RangeError('delta must lie in (0, 1)');
    return (Math.sqrt(2 * Math.log(1.25 / delta)) * sensitivity) / epsilon;
  }

  /**
   * Applies the Gaussian mechanism to a weight-delta vector before transmission.
   * Clips first (if `clipNorm` is set) so a single client's contribution to the
   * aggregate is bounded, then adds i.i.d. N(0, sigma^2) noise per coordinate.
   */
  static addGaussianNoise(vec: Float32Array, config: DifferentialPrivacyConfig): NoiseResult {
    const clipped = config.clipNorm !== undefined ? this.clipByNorm(vec, config.clipNorm) : vec;
    const sensitivity = config.clipNorm !== undefined ? 2 * config.clipNorm : config.sensitivity;
    const sigma = this.gaussianSigma(config.epsilon, config.delta, sensitivity);

    const noised = new Float32Array(clipped.length);
    for (let i = 0; i < clipped.length; i += 2) {
      const [z0, z1] = secureGaussianPair();
      noised[i] = (clipped[i] ?? 0) + z0 * sigma;
      if (i + 1 < clipped.length) {
        noised[i + 1] = (clipped[i + 1] ?? 0) + z1 * sigma;
      }
    }
    return { noised, sigma };
  }

  /** Pure epsilon-DP alternative via the Laplace mechanism, for pure-DP requirements (delta = 0). */
  static addLaplaceNoise(vec: Float32Array, epsilon: number, sensitivity: number): Float32Array {
    if (epsilon <= 0) throw new RangeError('epsilon must be > 0');
    const scale = sensitivity / epsilon;
    const out = new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
      const u = secureUniform() - 0.5;
      const noise = -scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
      out[i] = (vec[i] ?? 0) + noise;
    }
    return out;
  }
}
