import assert from 'node:assert/strict';
import { test } from 'node:test';
import { PrivacyEngine } from './PrivacyEngine.js';

test('l2Norm computes the Euclidean norm', () => {
  const norm = PrivacyEngine.l2Norm(new Float32Array([3, 4]));
  assert.ok(Math.abs(norm - 5) < 1e-6);
});

test('clipByNorm leaves vectors under the bound untouched', () => {
  const vec = new Float32Array([1, 0, 0]);
  const clipped = PrivacyEngine.clipByNorm(vec, 5);
  assert.deepEqual(Array.from(clipped), [1, 0, 0]);
});

test('clipByNorm rescales vectors over the bound to exactly maxNorm', () => {
  const vec = new Float32Array([3, 4]); // norm 5
  const clipped = PrivacyEngine.clipByNorm(vec, 1);
  assert.ok(Math.abs(PrivacyEngine.l2Norm(clipped) - 1) < 1e-5);
});

test('gaussianSigma matches the analytic Gaussian mechanism formula', () => {
  const epsilon = 1;
  const delta = 1e-5;
  const sensitivity = 2;
  const sigma = PrivacyEngine.gaussianSigma(epsilon, delta, sensitivity);
  const expected = (Math.sqrt(2 * Math.log(1.25 / delta)) * sensitivity) / epsilon;
  assert.ok(Math.abs(sigma - expected) < 1e-9);
});

test('gaussianSigma rejects a non-positive epsilon', () => {
  assert.throws(() => PrivacyEngine.gaussianSigma(0, 1e-5, 1), RangeError);
  assert.throws(() => PrivacyEngine.gaussianSigma(-1, 1e-5, 1), RangeError);
});

test('gaussianSigma rejects delta outside (0, 1)', () => {
  assert.throws(() => PrivacyEngine.gaussianSigma(1, 0, 1), RangeError);
  assert.throws(() => PrivacyEngine.gaussianSigma(1, 1, 1), RangeError);
});

test('addGaussianNoise clips before noising when clipNorm is set, using 2*clipNorm as sensitivity', () => {
  const vec = new Float32Array([30, 40]); // norm 50, will be clipped to norm 2
  const { sigma } = PrivacyEngine.addGaussianNoise(vec, { epsilon: 1, delta: 1e-5, sensitivity: 1, clipNorm: 2 });
  const expectedSigma = PrivacyEngine.gaussianSigma(1, 1e-5, 4); // sensitivity = 2*clipNorm
  assert.ok(Math.abs(sigma - expectedSigma) < 1e-9);
});

test('addGaussianNoise actually perturbs every coordinate', () => {
  const vec = new Float32Array(16).fill(1);
  const { noised } = PrivacyEngine.addGaussianNoise(vec, { epsilon: 1, delta: 1e-5, sensitivity: 1 });
  assert.equal(noised.length, vec.length);
  // With sigma > 0 the odds every single coordinate lands back on exactly 1 are negligible.
  assert.ok(Array.from(noised).some((v) => v !== 1));
});

test('addLaplaceNoise perturbs every coordinate and rejects non-positive epsilon', () => {
  const vec = new Float32Array(8).fill(0);
  const noised = PrivacyEngine.addLaplaceNoise(vec, 1, 1);
  assert.equal(noised.length, vec.length);
  assert.ok(Array.from(noised).some((v) => v !== 0));
  assert.throws(() => PrivacyEngine.addLaplaceNoise(vec, 0, 1), RangeError);
});
