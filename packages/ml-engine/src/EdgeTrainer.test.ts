import { copyFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { EdgeTrainer, readCheckpoint, writeCheckpoint, type EdgeCheckpoint } from './EdgeTrainer.js';
import type { DataSource, LocalDataset } from './types.js';

// Fixtures live outside src/ (they're binary, not TypeScript) at
// packages/ml-engine/fixtures - one directory up from this compiled test's
// own location (dist-test/EdgeTrainer.test.js), regardless of whether that
// compile target is dist-test/ or dist/.
const FIXTURE_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'fixtures');
const BACKBONE_FIXTURE = path.join(FIXTURE_DIR, 'linear-relu-8x8.onnx');

const SHAPE = { inputDim: 4, outputDim: 2 };

function zeroCheckpoint(): EdgeCheckpoint {
  return {
    inputDim: SHAPE.inputDim,
    outputDim: SHAPE.outputDim,
    weights: new Float32Array(SHAPE.inputDim * SHAPE.outputDim),
    bias: new Float32Array(SHAPE.outputDim),
  };
}

test('writeCheckpoint / readCheckpoint round-trip a HeadParameters shape', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'nexus-ppfl-test-'));
  const file = path.join(dir, 'checkpoint.json');
  const checkpoint: EdgeCheckpoint = {
    inputDim: 3,
    outputDim: 2,
    weights: new Float32Array([1, 2, 3, 4, 5, 6]),
    bias: new Float32Array([0.5, -0.5]),
  };

  await writeCheckpoint(file, checkpoint);
  const restored = await readCheckpoint(file);

  assert.equal(restored.inputDim, 3);
  assert.equal(restored.outputDim, 2);
  assert.deepEqual(Array.from(restored.weights), Array.from(checkpoint.weights));
  assert.deepEqual(Array.from(restored.bias), Array.from(checkpoint.bias));
});

test('EdgeTrainer: with a real model.onnx present, trains on the backbone-produced embedding', async () => {
  // Exercises the InferenceSession-backed path in trainEpoch()/runBackbone()
  // - a real, tiny (543-byte) linear+ReLU ONNX model, not the no-backbone
  // fallback every other test in this file uses. The fixture's raw input
  // AND its embedding output are both 8-dimensional (see
  // fixtures/generate_model.py) - loadLocalData sizes raw features to the
  // checkpoint's inputDim, and that same value doubles as the head's
  // expected embedding size, so both must be 8 here (unlike SHAPE, the
  // 4-dim checkpoint the other tests in this file use for the no-backbone
  // path, where raw features ARE the embedding directly).
  const backboneShape = { inputDim: 8, outputDim: 2 };
  const dir = await mkdtemp(path.join(tmpdir(), 'nexus-ppfl-test-backbone-'));
  await copyFile(BACKBONE_FIXTURE, path.join(dir, 'model.onnx'));
  await writeCheckpoint(path.join(dir, 'checkpoint.json'), {
    inputDim: backboneShape.inputDim,
    outputDim: backboneShape.outputDim,
    weights: new Float32Array(backboneShape.inputDim * backboneShape.outputDim),
    bias: new Float32Array(backboneShape.outputDim),
  });

  const trainer = new EdgeTrainer();
  await trainer.loadArtifacts(dir);
  await trainer.loadLocalData('sqlite:///backbone-test-client.db', { numSamples: 32 });
  const report = await trainer.trainEpoch(8, 0.1);

  assert.equal(report.numSamples, 32);
  assert.ok(Number.isFinite(report.avgLoss));

  const delta = trainer.extractWeightDeltas();
  assert.equal(delta.length, backboneShape.inputDim * backboneShape.outputDim + backboneShape.outputDim);
  assert.ok(Array.from(delta).some((v) => v !== 0), 'weight delta should not be all zeros after training');

  await trainer.dispose();
});

test('EdgeTrainer: loadArtifacts without a model.onnx trains directly on raw features', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'nexus-ppfl-test-'));
  await writeCheckpoint(path.join(dir, 'checkpoint.json'), zeroCheckpoint());

  const trainer = new EdgeTrainer();
  await trainer.loadArtifacts(dir);
  await trainer.loadLocalData('sqlite:///unit-test-client.db', { numSamples: 32 });
  const report = await trainer.trainEpoch(8, 0.1);

  assert.equal(report.numSamples, 32);
  assert.equal(report.batches, 4);
  assert.ok(Number.isFinite(report.avgLoss));

  const delta = trainer.extractWeightDeltas();
  assert.equal(delta.length, SHAPE.inputDim * SHAPE.outputDim + SHAPE.outputDim);
  assert.ok(Array.from(delta).some((v) => v !== 0), 'weight delta should not be all zeros after training');

  await trainer.dispose();
});

test('EdgeTrainer: loss decreases across successive epochs on the same synthetic data', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'nexus-ppfl-test-'));
  await writeCheckpoint(path.join(dir, 'checkpoint.json'), zeroCheckpoint());

  const trainer = new EdgeTrainer();
  await trainer.loadArtifacts(dir);
  await trainer.loadLocalData('sqlite:///convergence-test.db', { numSamples: 64 });

  const losses: number[] = [];
  for (let epoch = 0; epoch < 5; epoch++) {
    const report = await trainer.trainEpoch(16, 0.2);
    losses.push(report.avgLoss);
  }
  await trainer.dispose();

  assert.ok(losses[0]! > losses.at(-1)!, `loss should trend down: ${losses.join(', ')}`);
});

test('EdgeTrainer: loadLocalData is deterministic for a given dbPath (seeded RNG)', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'nexus-ppfl-test-'));
  await writeCheckpoint(path.join(dir, 'checkpoint.json'), zeroCheckpoint());

  const trainerA = new EdgeTrainer();
  await trainerA.loadArtifacts(dir);
  const datasetA = await trainerA.loadLocalData('sqlite:///same-client.db', { numSamples: 16 });
  await trainerA.dispose();

  const trainerB = new EdgeTrainer();
  await trainerB.loadArtifacts(dir);
  const datasetB = await trainerB.loadLocalData('sqlite:///same-client.db', { numSamples: 16 });
  await trainerB.dispose();

  assert.deepEqual(Array.from(datasetA.features), Array.from(datasetB.features));
  assert.deepEqual(Array.from(datasetA.labels), Array.from(datasetB.labels));
});

test('EdgeTrainer: loadDataFrom accepts a custom DataSource', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'nexus-ppfl-test-'));
  await writeCheckpoint(path.join(dir, 'checkpoint.json'), zeroCheckpoint());

  const fixedDataset: LocalDataset = {
    features: new Float32Array(SHAPE.inputDim * 4).fill(1),
    labels: new Float32Array([0, 1, 0, 1]),
    numSamples: 4,
    inputDim: SHAPE.inputDim,
  };
  const source: DataSource = { load: async () => fixedDataset };

  const trainer = new EdgeTrainer();
  await trainer.loadArtifacts(dir);
  const loaded = await trainer.loadDataFrom(source);
  assert.strictEqual(loaded, fixedDataset);

  const report = await trainer.trainEpoch(4);
  assert.equal(report.numSamples, 4);
  await trainer.dispose();
});

test('EdgeTrainer: methods called out of order throw a clear error', async () => {
  const trainer = new EdgeTrainer();
  await assert.rejects(() => trainer.loadLocalData('sqlite:///x.db'));
  await assert.rejects(() => trainer.trainEpoch(8));
});
