import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NexusNetworkNode, TOPICS } from './NexusNetworkNode.js';

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Publishing before the gossipsub mesh has formed can silently drop a
 * message (no mesh peers yet to flood to), so this polls with repeated
 * publishes instead of a single fixed sleep - it waits only as long as the
 * mesh actually takes to form, rather than a guessed fixed delay, without
 * ever being flaky about *whether* delivery eventually happens.
 */
async function publishUntilDelivered(
  sender: NexusNetworkNode,
  topic: (typeof TOPICS)[keyof typeof TOPICS],
  payload: Uint8Array,
  isDelivered: () => boolean,
  timeoutMs: number,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!isDelivered() && Date.now() < deadline) {
    await sender.publish(topic, payload);
    await delay(250);
  }
}

test('onMessage() registered before start() still receives messages once started', async () => {
  const nodeA = new NexusNetworkNode({ role: 'aggregator', listenAddresses: ['/ip4/127.0.0.1/tcp/0/ws'] });

  let received: Uint8Array | undefined;
  // This is the exact ordering that used to crash (see NexusNetworkNode's
  // history): onMessage() called before start(). It must not throw, and the
  // handler must still fire once both nodes are up and connected.
  nodeA.onMessage(TOPICS.WEIGHT_UPDATES, (data) => {
    received = data;
  });

  await nodeA.start();

  try {
    const nodeAAddr = nodeA.multiaddrs.find((addr) => addr.includes('/ws'));
    assert.ok(nodeAAddr, 'nodeA should expose a websocket multiaddr');

    const nodeB = new NexusNetworkNode({ role: 'edge-client', bootstrapPeers: [nodeAAddr] });
    await nodeB.start();

    try {
      const payload = new Uint8Array([1, 2, 3, 4]);
      await publishUntilDelivered(nodeB, TOPICS.WEIGHT_UPDATES, payload, () => received !== undefined, 10_000);

      assert.ok(received, 'nodeA never received the message published by nodeB');
      assert.deepEqual(Array.from(received!), Array.from(payload));
    } finally {
      await nodeB.stop();
    }
  } finally {
    await nodeA.stop();
  }
});

test('two nodes with zero shared bootstrap config discover and connect purely via mDNS', async () => {
  const nodeA = new NexusNetworkNode({ role: 'aggregator', listenAddresses: ['/ip4/0.0.0.0/tcp/0/ws'] });
  const nodeB = new NexusNetworkNode({ role: 'edge-client', listenAddresses: ['/ip4/0.0.0.0/tcp/0/ws'] });

  let received: Uint8Array | undefined;
  nodeA.onMessage(TOPICS.WEIGHT_UPDATES, (data) => {
    received = data;
  });

  await nodeA.start();
  await nodeB.start();

  try {
    const payload = new Uint8Array([42]);
    // mDNS discovery + auto-dial can take a few announce intervals; this
    // regression-tests that NexusNetworkNode dials on peer:discovery itself
    // (libp2p does not do this automatically) rather than merely knowing
    // the peer exists.
    await publishUntilDelivered(nodeB, TOPICS.WEIGHT_UPDATES, payload, () => received !== undefined, 15_000);
    assert.ok(received, 'nodeA never received the message published by nodeB over pure mDNS discovery');
    assert.deepEqual(Array.from(received!), Array.from(payload));
  } finally {
    await nodeB.stop();
    await nodeA.stop();
  }
});

test('multiple onMessage() handlers on the same topic all fire', async () => {
  const nodeA = new NexusNetworkNode({ role: 'aggregator', listenAddresses: ['/ip4/127.0.0.1/tcp/0/ws'] });
  await nodeA.start();

  let firstCount = 0;
  let secondCount = 0;
  nodeA.onMessage(TOPICS.ROUND_CONTROL, () => {
    firstCount += 1;
  });
  nodeA.onMessage(TOPICS.ROUND_CONTROL, () => {
    secondCount += 1;
  });

  try {
    const nodeAAddr = nodeA.multiaddrs.find((addr) => addr.includes('/ws'));
    assert.ok(nodeAAddr);
    const nodeB = new NexusNetworkNode({ role: 'edge-client', bootstrapPeers: [nodeAAddr] });
    await nodeB.start();

    try {
      await publishUntilDelivered(
        nodeB,
        TOPICS.ROUND_CONTROL,
        new Uint8Array([9]),
        () => firstCount > 0 && secondCount > 0,
        10_000,
      );
      assert.equal(firstCount, 1);
      assert.equal(secondCount, 1);
    } finally {
      await nodeB.stop();
    }
  } finally {
    await nodeA.stop();
  }
});
