import assert from 'node:assert/strict';
import { test } from 'node:test';
import { WeightSerializer, NexusNetworkNode, TOPICS } from '@nexus-ppfl/network';
import { AggregatorNode } from './AggregatorNode.js';

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

test('FedAvg averages client deltas and adds the mean to the global weights', () => {
  const global = new Float32Array([100, 200]);
  const deltas = [new Float32Array([1, -1]), new Float32Array([3, 1]), new Float32Array([2, 0])];
  const result = AggregatorNode.FedAvg(global, deltas);
  // mean delta = [(1+3+2)/3, (-1+1+0)/3] = [2, 0]
  assert.ok(Math.abs(result[0]! - 102) < 1e-5);
  assert.ok(Math.abs(result[1]! - 200) < 1e-5);
});

test('FedAvg does not mutate its inputs', () => {
  const global = new Float32Array([1, 1]);
  const delta = new Float32Array([5, 5]);
  AggregatorNode.FedAvg(global, [delta]);
  assert.deepEqual(Array.from(global), [1, 1]);
  assert.deepEqual(Array.from(delta), [5, 5]);
});

test('FedAvg with zero clients returns a copy of the global weights unchanged', () => {
  const global = new Float32Array([1, 2, 3]);
  const result = AggregatorNode.FedAvg(global, []);
  assert.deepEqual(Array.from(result), [1, 2, 3]);
  assert.notStrictEqual(result, global);
});

test('FedAvg rejects a delta whose length does not match the global model', () => {
  const global = new Float32Array([1, 2, 3]);
  assert.throws(() => AggregatorNode.FedAvg(global, [new Float32Array([1, 2])]));
});

test('AggregatorNode.waitForThreshold resolves once k distinct clients report, ignoring duplicates and stale rounds', async () => {
  const aggregator = new AggregatorNode({ listenAddresses: ['/ip4/127.0.0.1/tcp/0/ws'] });
  await aggregator.start();

  const serializer = new WeightSerializer();
  const aggregatorAddr = aggregator.multiaddrs.find((addr) => addr.includes('/ws'));
  assert.ok(aggregatorAddr);

  const client = new NexusNetworkNode({ role: 'edge-client', bootstrapPeers: [aggregatorAddr] });
  await client.start();

  try {
    const waitPromise = aggregator.waitForThreshold(2, 15_000);

    const publishUpdate = async (clientId: string, round: number, value: number) => {
      const payload = await serializer.serializeUpdate({
        clientId,
        round,
        numSamples: 10,
        weights: new Float32Array([value]),
      });
      await client.publish(TOPICS.WEIGHT_UPDATES, payload);
    };

    // Poll-publish until the aggregator has what it needs, tolerating mesh
    // formation latency without a flaky fixed sleep.
    const deadline = Date.now() + 15_000;
    while (aggregator.receivedCount < 2 && Date.now() < deadline) {
      await publishUpdate('stale-client', 999, -1); // wrong round: must be ignored
      await publishUpdate('client-a', 0, 1);
      await publishUpdate('client-a', 0, 1); // duplicate clientId/round: must be deduped
      await publishUpdate('client-b', 0, 2);
      await delay(250);
    }

    const deltas = await waitPromise;
    assert.equal(deltas.length, 2);
    const values = deltas.map((d) => d[0]).sort();
    assert.deepEqual(values, [1, 2]);
  } finally {
    await client.stop();
    await aggregator.stop();
  }
});

test('AggregatorNode.waitForThreshold resolves with fewer than k on timeout', async () => {
  const aggregator = new AggregatorNode({ listenAddresses: ['/ip4/127.0.0.1/tcp/0/ws'] });
  await aggregator.start();
  try {
    const deltas = await aggregator.waitForThreshold(5, 200);
    assert.equal(deltas.length, 0);
  } finally {
    await aggregator.stop();
  }
});
