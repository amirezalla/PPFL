import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  LinearHead,
  cloneHeadParameters,
  flattenHeadParameters,
  subtractFlat,
  unflattenHeadParameters,
  type HeadParameters,
} from './LinearHead.js';

function makeSeparableBatch(inputDim: number, outputDim: number, samplesPerClass: number) {
  const numSamples = outputDim * samplesPerClass;
  const features = new Float32Array(numSamples * inputDim);
  const labels = new Float32Array(numSamples);
  let n = 0;
  for (let label = 0; label < outputDim; label++) {
    for (let s = 0; s < samplesPerClass; s++) {
      labels[n] = label;
      for (let i = 0; i < inputDim; i++) {
        // A one-hot-ish blob per class: linearly separable, so SGD should converge cleanly.
        features[n * inputDim + i] = i === label ? 2 : -0.1;
      }
      n += 1;
    }
  }
  return { features, labels, numSamples };
}

test('LinearHead.trainStep reduces loss on a linearly separable batch', () => {
  const inputDim = 4;
  const outputDim = 3;
  const params: HeadParameters = {
    inputDim,
    outputDim,
    weights: new Float32Array(inputDim * outputDim),
    bias: new Float32Array(outputDim),
  };
  const head = new LinearHead(params);
  const { features, labels, numSamples } = makeSeparableBatch(inputDim, outputDim, 8);

  const losses: number[] = [];
  for (let step = 0; step < 20; step++) {
    losses.push(head.trainStep(features, numSamples, labels, 0.5));
  }

  assert.ok(losses[0]! > losses[losses.length - 1]!, `loss should decrease: ${losses[0]} -> ${losses.at(-1)}`);
  assert.ok(losses.at(-1)! < 0.2, `final loss should be small on a separable batch, got ${losses.at(-1)}`);
});

test('cloneHeadParameters produces an independent copy', () => {
  const original: HeadParameters = {
    inputDim: 2,
    outputDim: 2,
    weights: new Float32Array([1, 2, 3, 4]),
    bias: new Float32Array([5, 6]),
  };
  const clone = cloneHeadParameters(original);
  clone.weights[0] = 999;
  assert.equal(original.weights[0], 1);
});

test('flattenHeadParameters / unflattenHeadParameters round-trip', () => {
  const params: HeadParameters = {
    inputDim: 3,
    outputDim: 2,
    weights: new Float32Array([1, 2, 3, 4, 5, 6]),
    bias: new Float32Array([7, 8]),
  };
  const flat = flattenHeadParameters(params);
  assert.equal(flat.length, 8);
  const restored = unflattenHeadParameters(flat, 3, 2);
  assert.deepEqual(Array.from(restored.weights), Array.from(params.weights));
  assert.deepEqual(Array.from(restored.bias), Array.from(params.bias));
});

test('subtractFlat computes element-wise difference', () => {
  const a = new Float32Array([5, 5, 5]);
  const b = new Float32Array([1, 2, 3]);
  assert.deepEqual(Array.from(subtractFlat(a, b)), [4, 3, 2]);
});
