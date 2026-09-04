import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { PrivacyEngine } from '@nexus-ppfl/crypto';
import { NexusNetworkNode, TOPICS, encodeWeightUpdate } from '@nexus-ppfl/network';
import { EdgeTrainer, writeCheckpoint, type EdgeCheckpoint } from './EdgeTrainer.js';
import { DEFAULT_DP_CONFIG, MODEL_SHAPE } from './modelShape.js';

export {
  EdgeTrainer,
  readCheckpoint,
  writeCheckpoint,
  type EdgeCheckpoint,
} from './EdgeTrainer.js';
export { CsvDataSource, type CsvDataSourceOptions } from './DataLoader.js';
export {
  LinearHead,
  cloneHeadParameters,
  flattenHeadParameters,
  unflattenHeadParameters,
  subtractFlat,
  type HeadParameters,
} from './LinearHead.js';
export { AggregatorNode, type AggregatorNodeOptions } from './AggregatorNode.js';
export { runAggregatorServer, type AggregatorServerConfig, type AggregatorReadyMessage } from './server.js';
export { runEdgeClient, type EdgeClientOptions, type EdgeClientHandle } from './client.js';
export { MODEL_SHAPE, MODEL_LENGTH, DEFAULT_DP_CONFIG } from './modelShape.js';
export type { DataSource, LocalDataset, EpochReport } from './types.js';

/**
 * End-to-end local cycle: bootstraps a zero-initialized global checkpoint,
 * trains one `EdgeTrainer` epoch against synthetic non-IID data, applies
 * Phase 1's Gaussian differential-privacy mechanism to the resulting weight
 * delta, and publishes the noised update over a live libp2p PubSub node -
 * Phase 1's `NexusNetworkNode`, started locally just for this run. Not
 * auto-executed on import; call it explicitly, or run this module directly
 * (`node dist/index.js`) to see it end to end.
 */
export async function runEndToEndDemo(clientId = 'edge-client-demo'): Promise<void> {
  const { inputDim, outputDim } = MODEL_SHAPE;
  const baseline: EdgeCheckpoint = {
    inputDim,
    outputDim,
    weights: new Float32Array(inputDim * outputDim),
    bias: new Float32Array(outputDim),
  };

  const artifactDir = await mkdtemp(path.join(tmpdir(), 'nexus-ppfl-'));
  await writeCheckpoint(path.join(artifactDir, 'checkpoint.json'), baseline);
  // No model.onnx dropped into artifactDir: this run exercises the no-backbone
  // (plain logistic regression) path. Ship a real frozen ONNX feature
  // extractor as "model.onnx" alongside checkpoint.json to exercise the
  // InferenceSession-backed embedding path instead.

  const trainer = new EdgeTrainer();
  await trainer.loadArtifacts(artifactDir);
  await trainer.loadLocalData(`sqlite:///${clientId}.db`, { numSamples: 128 });
  const report = await trainer.trainEpoch(16);
  console.log(
    `[EdgeTrainer] trained ${report.numSamples} samples over ${report.batches} batches, avg loss ${report.avgLoss.toFixed(4)}`,
  );

  const rawDelta = trainer.extractWeightDeltas();
  await trainer.dispose();

  const { noised: privateDelta, sigma } = PrivacyEngine.addGaussianNoise(rawDelta, DEFAULT_DP_CONFIG);
  console.log(`[PrivacyEngine] applied Gaussian mechanism, sigma=${sigma.toFixed(4)}`);

  const node = new NexusNetworkNode({ role: 'edge-client' });
  await node.start();
  try {
    const payload = await encodeWeightUpdate({
      clientId,
      round: 0,
      numSamples: report.numSamples,
      weights: privateDelta,
    });
    await node.publish(TOPICS.WEIGHT_UPDATES, payload);
    console.log(
      `[NexusNetworkNode] published ${payload.byteLength}-byte WeightUpdate on ${TOPICS.WEIGHT_UPDATES} as ${node.peerId}`,
    );
  } finally {
    await node.stop();
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  runEndToEndDemo().catch((err: unknown) => {
    console.error(err);
    process.exitCode = 1;
  });
}
