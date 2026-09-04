import assert from 'node:assert/strict';
import { test } from 'node:test';
import { FederatedAggregationRound, federatedAverage } from './Aggregator.js';

test('federatedAverage weights each client by its sample count', () => {
  const result = federatedAverage([
    { clientId: 'a', weights: new Float32Array([10, 0]), numSamples: 1 },
    { clientId: 'b', weights: new Float32Array([0, 10]), numSamples: 3 },
  ]);
  // total 4 samples: a contributes 1/4 * [10,0], b contributes 3/4 * [0,10]
  assert.ok(Math.abs(result[0]! - 2.5) < 1e-5);
  assert.ok(Math.abs(result[1]! - 7.5) < 1e-5);
});

test('federatedAverage rejects mismatched weight-vector lengths', () => {
  assert.throws(() =>
    federatedAverage([
      { clientId: 'a', weights: new Float32Array([1, 2]), numSamples: 1 },
      { clientId: 'b', weights: new Float32Array([1, 2, 3]), numSamples: 1 },
    ]),
  );
});

test('federatedAverage rejects an empty update set', () => {
  assert.throws(() => federatedAverage([]));
});

test('FederatedAggregationRound settles early once the completion threshold is met', async () => {
  const round = new FederatedAggregationRound({ expectedClients: 5, completionThreshold: 0.8, roundTimeoutMs: 5000 });
  for (let i = 0; i < 5; i++) round.registerExpectedClient(`c${i}`);

  const resultPromise = round.start();
  for (let i = 0; i < 4; i++) {
    round.submitUpdate({ clientId: `c${i}`, weights: new Float32Array([i]), numSamples: 1 });
  }

  const result = await resultPromise;
  assert.equal(result.participantCount, 4);
  assert.deepEqual(result.droppedStragglers, ['c4']);
});

test('FederatedAggregationRound settles on timeout with fewer than the threshold', async () => {
  const round = new FederatedAggregationRound({ expectedClients: 5, completionThreshold: 0.8, roundTimeoutMs: 50 });
  for (let i = 0; i < 5; i++) round.registerExpectedClient(`c${i}`);

  const resultPromise = round.start();
  round.submitUpdate({ clientId: 'c0', weights: new Float32Array([1]), numSamples: 1 });

  const result = await resultPromise;
  assert.equal(result.participantCount, 1);
  assert.equal(result.droppedStragglers.length, 4);
});

test('FederatedAggregationRound ignores submissions after settling', async () => {
  const round = new FederatedAggregationRound({ expectedClients: 2, completionThreshold: 1, roundTimeoutMs: 5000 });
  round.registerExpectedClient('a');
  round.registerExpectedClient('b');

  const resultPromise = round.start();
  round.submitUpdate({ clientId: 'a', weights: new Float32Array([1]), numSamples: 1 });
  round.submitUpdate({ clientId: 'b', weights: new Float32Array([2]), numSamples: 1 });
  const result = await resultPromise;

  // A late submission after the round already settled must not affect the result.
  round.submitUpdate({ clientId: 'c', weights: new Float32Array([999]), numSamples: 1 });
  assert.equal(result.participantCount, 2);
});
