import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import assert from 'node:assert/strict';
import path from 'node:path';
import { test } from 'node:test';
import { PrivacyEngine } from '@nexus-ppfl/crypto';
import { AggregatorNode } from './AggregatorNode.js';
import { EdgeTrainer, writeCheckpoint } from './EdgeTrainer.js';
import { unflattenHeadParameters } from './LinearHead.js';
import { DEFAULT_DP_CONFIG, MODEL_LENGTH, MODEL_SHAPE } from './modelShape.js';

/**
 * Regression test for a real bug found while running the multi-process
 * cluster test: the original demo defaults (epsilon=2, clipNorm=1.0)
 * produced a Gaussian noise sigma about 20x larger than the model's actual
 * gradient L2 norm (~0.24), so FedAvg's running sum was pure noise and
 * client loss exploded from ~1.0 to ~11 after a single round instead of
 * decreasing. This simulates several federated rounds end to end with the
 * shipped DEFAULT_DP_CONFIG and asserts loss stays bounded - if someone
 * changes epsilon/clipNorm back to a badly calibrated pair, this fails
 * instead of only showing up as a confusing multi-process test run.
 */
test('DEFAULT_DP_CONFIG keeps client loss bounded across several federated rounds', async () => {
  const numClients = 2;
  const numRounds = 5;
  let globalWeights: Float32Array = new Float32Array(MODEL_LENGTH);
  const roundLosses: number[] = [];

  for (let round = 0; round < numRounds; round++) {
    const baseline = unflattenHeadParameters(globalWeights, MODEL_SHAPE.inputDim, MODEL_SHAPE.outputDim);
    const deltas: Float32Array[] = [];
    let lossSum = 0;

    for (let c = 0; c < numClients; c++) {
      const dir = await mkdtemp(path.join(tmpdir(), `dp-cal-r${round}-c${c}-`));
      await writeCheckpoint(path.join(dir, 'checkpoint.json'), baseline);

      const trainer = new EdgeTrainer();
      await trainer.loadArtifacts(dir);
      await trainer.loadLocalData(`sqlite:///dp-cal-client-${c}.db`, { numSamples: 64 });
      const report = await trainer.trainEpoch(16);
      lossSum += report.avgLoss;

      const rawDelta = trainer.extractWeightDeltas();
      await trainer.dispose();

      const { noised } = PrivacyEngine.addGaussianNoise(rawDelta, DEFAULT_DP_CONFIG);
      deltas.push(noised);
    }

    globalWeights = AggregatorNode.FedAvg(globalWeights, deltas);
    roundLosses.push(lossSum / numClients);
  }

  for (const loss of roundLosses) {
    assert.ok(
      loss < 3,
      `avg client loss ${loss} exceeded the sane bound of 3 - DP noise may again be dominating the real signal (all losses: ${roundLosses.join(', ')})`,
    );
  }

  const finalNorm = PrivacyEngine.l2Norm(globalWeights);
  assert.ok(
    finalNorm < 20,
    `global weight L2 norm ${finalNorm} grew unreasonably large after ${numRounds} rounds - likely a noise random walk again dominating FedAvg`,
  );
});

test('DEFAULT_DP_CONFIG sigma is the same order of magnitude as this model\'s real gradient norm', () => {
  // A cruder, instant sanity check on the constant itself (no training
  // involved): the empirically observed raw delta L2 norm for this toy
  // model is ~0.24 - sigma should not be wildly (>5x) larger than that.
  const sigma = PrivacyEngine.gaussianSigma(
    DEFAULT_DP_CONFIG.epsilon,
    DEFAULT_DP_CONFIG.delta,
    2 * DEFAULT_DP_CONFIG.clipNorm,
  );
  const observedRawDeltaNorm = 0.24;
  assert.ok(
    sigma < observedRawDeltaNorm * 5,
    `sigma=${sigma} is more than 5x the observed raw delta norm ${observedRawDeltaNorm} - noise would dominate signal`,
  );
});
