import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { PrivacyEngine } from '@nexus-ppfl/crypto';
import { NexusNetworkNode, TOPICS, WeightSerializer } from '@nexus-ppfl/network';
import { EdgeTrainer, writeCheckpoint, type EdgeCheckpoint } from './EdgeTrainer.js';
import { unflattenHeadParameters } from './LinearHead.js';
import { DEFAULT_DP_CONFIG, MODEL_SHAPE } from './modelShape.js';

export interface EdgeClientOptions {
  clientId: string;
  /** The aggregator's multiaddr to bootstrap-dial, e.g. from AggregatorNode.multiaddrs. */
  aggregatorAddr: string;
  numSamples?: number;
  batchSize?: number;
  dpEpsilon?: number;
}

export interface EdgeClientHandle {
  stop: () => Promise<void>;
}

/**
 * Runs one edge client for the lifetime of the process: connects to the
 * aggregator, then on every RoundControl(ROUND_START) trains one local
 * epoch against synthetic non-IID data (Phase 2's EdgeTrainer), applies the
 * Gaussian DP mechanism (Phase 1's PrivacyEngine), and publishes the
 * resulting delta as a WeightUpdate - closing the loop the aggregator's
 * FedAvg (Phase 3) consumes. Tracks the most recent GlobalModel broadcast
 * as next round's training baseline.
 */
export async function runEdgeClient(options: EdgeClientOptions): Promise<EdgeClientHandle> {
  const { clientId, aggregatorAddr } = options;
  const numSamples = options.numSamples ?? 64;
  const batchSize = options.batchSize ?? 16;
  const dpEpsilon = options.dpEpsilon ?? DEFAULT_DP_CONFIG.epsilon;

  const serializer = new WeightSerializer();
  const node = new NexusNetworkNode({ role: 'edge-client', bootstrapPeers: [aggregatorAddr] });

  let baseline: EdgeCheckpoint = {
    inputDim: MODEL_SHAPE.inputDim,
    outputDim: MODEL_SHAPE.outputDim,
    weights: new Float32Array(MODEL_SHAPE.inputDim * MODEL_SHAPE.outputDim),
    bias: new Float32Array(MODEL_SHAPE.outputDim),
  };

  async function runRound(round: number): Promise<void> {
    const artifactDir = await mkdtemp(path.join(tmpdir(), `nexus-ppfl-${clientId}-`));
    await writeCheckpoint(path.join(artifactDir, 'checkpoint.json'), baseline);

    const trainer = new EdgeTrainer();
    await trainer.loadArtifacts(artifactDir);
    await trainer.loadLocalData(`sqlite:///${clientId}.db`, { numSamples });
    const report = await trainer.trainEpoch(batchSize);
    const rawDelta = trainer.extractWeightDeltas();
    await trainer.dispose();

    const { noised: privateDelta } = PrivacyEngine.addGaussianNoise(rawDelta, {
      epsilon: dpEpsilon,
      delta: DEFAULT_DP_CONFIG.delta,
      sensitivity: DEFAULT_DP_CONFIG.sensitivity,
      clipNorm: DEFAULT_DP_CONFIG.clipNorm,
    });

    const payload = await serializer.serializeUpdate({
      clientId,
      round,
      numSamples: report.numSamples,
      weights: privateDelta,
    });
    await node.publish(TOPICS.WEIGHT_UPDATES, payload);
    console.log(
      `[${clientId}] round ${round}: trained ${report.numSamples} samples (avg loss ${report.avgLoss.toFixed(4)}), published delta`,
    );
  }

  // onMessage() requires the libp2p node to already be running (it attaches
  // to node.services.pubsub), so start() must come first.
  await node.start();
  console.log(`[${clientId}] connected as ${node.peerId}, bootstrapped to ${aggregatorAddr}`);

  node.onMessage(TOPICS.ROUND_CONTROL, (data) => {
    void (async () => {
      const message = await serializer.deserializeRoundControl(data);
      if (message.phase !== 'ROUND_START') return;
      console.log(`[${clientId}] round ${message.round} started`);
      try {
        await runRound(message.round);
      } catch (err) {
        console.error(`[${clientId}] round ${message.round} failed:`, err);
      }
    })();
  });

  node.onMessage(TOPICS.GLOBAL_MODEL, (data) => {
    void (async () => {
      const message = await serializer.deserializeGlobalModel(data);
      baseline = unflattenHeadParameters(message.weights, MODEL_SHAPE.inputDim, MODEL_SHAPE.outputDim);
      console.log(
        `[${clientId}] received global model for round ${message.round} (hash ${message.modelHash.slice(0, 8)}...)`,
      );
    })();
  });

  return { stop: () => node.stop() };
}

function parseArgs(argv: string[]): { id: string; aggregator: string } {
  const parsed = new Map<string, string>();
  for (const arg of argv) {
    if (!arg.startsWith('--')) continue;
    const eq = arg.indexOf('=');
    if (eq === -1) continue;
    parsed.set(arg.slice(2, eq), arg.slice(eq + 1));
  }
  const id = parsed.get('id');
  const aggregator = parsed.get('aggregator');
  if (!id || !aggregator) {
    throw new Error('Usage: client.js --id=<clientId> --aggregator=<multiaddr>');
  }
  return { id, aggregator };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { id, aggregator } = parseArgs(process.argv.slice(2));
  runEdgeClient({ clientId: id, aggregatorAddr: aggregator })
    .then((handle) => {
      const shutdown = (): void => {
        void handle.stop().then(() => process.exit(0));
      };
      process.on('SIGINT', shutdown);
      process.on('SIGTERM', shutdown);
    })
    .catch((err: unknown) => {
      console.error(err);
      process.exit(1);
    });
}
