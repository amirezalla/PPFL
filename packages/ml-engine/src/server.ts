import process from 'node:process';
import { AggregatorNode } from './AggregatorNode.js';
import { MODEL_LENGTH, MODEL_SHAPE } from './modelShape.js';

export interface AggregatorServerConfig {
  /** Total parameter count of the model being aggregated. */
  modelLength: number;
  /** Number of client updates to wait for before running FedAvg. */
  clientsPerRound: number;
  /** How long to tolerate stragglers before closing a round early. */
  roundTimeoutMs: number;
  listenAddresses?: string[];
  /** Stop after this many rounds; omit to run forever. */
  maxRounds?: number;
  /**
   * Delay before round 0's broadcast, giving clients time to be launched,
   * bootstrap-dial, and form their gossipsub mesh with this node. Without
   * it, round 0 in a freshly-started cluster reliably times out with zero
   * updates - clients don't exist yet when the broadcast goes out. Default
   * 0 (no delay) for programmatic/library use; the CLI entrypoint below
   * sets a real default for cluster/demo runs.
   */
  startupGraceMs?: number;
}

export interface AggregatorReadyMessage {
  type: 'aggregator-ready';
  peerId: string;
  multiaddrs: string[];
}

/**
 * Continuous aggregation loop: broadcast round start, wait for a threshold
 * of client updates (tolerating stragglers), FedAvg them into the global
 * model, broadcast the result, advance the round, repeat. Runs until
 * `config.maxRounds` is reached, or forever if omitted.
 */
export async function runAggregatorServer(config: AggregatorServerConfig): Promise<void> {
  const aggregator = new AggregatorNode({ listenAddresses: config.listenAddresses });
  await aggregator.start();
  console.log(`[AggregatorNode] listening as ${aggregator.peerId} on [${aggregator.multiaddrs.join(', ')}]`);

  // When launched via child_process.fork() (see scripts/test-cluster.ts),
  // process.send exists and lets the parent harness learn this node's
  // multiaddr without polling a file or hardcoding a port - it's undefined,
  // and this is a no-op, when run standalone.
  const readyMessage: AggregatorReadyMessage = {
    type: 'aggregator-ready',
    peerId: aggregator.peerId,
    multiaddrs: aggregator.multiaddrs,
  };
  process.send?.(readyMessage);

  if (config.startupGraceMs) {
    await new Promise((resolve) => setTimeout(resolve, config.startupGraceMs));
  }

  let globalWeights: Float32Array = new Float32Array(config.modelLength);

  try {
    while (config.maxRounds === undefined || aggregator.round < config.maxRounds) {
      const round = aggregator.round;

      await aggregator.broadcastRoundStart(round, config.clientsPerRound);
      console.log(
        `[Round ${round}] started, waiting for ${config.clientsPerRound} updates (timeout ${config.roundTimeoutMs}ms)`,
      );

      let clientDeltas = await aggregator.waitForThreshold(config.clientsPerRound, config.roundTimeoutMs);
      console.log(`[Round ${round}] received ${clientDeltas.length}/${config.clientsPerRound} updates`);

      if (clientDeltas.length === 0) {
        console.log(`[Round ${round}] no updates arrived before timeout; skipping FedAvg`);
        aggregator.advanceRound();
        continue;
      }

      globalWeights = AggregatorNode.FedAvg(globalWeights, clientDeltas);
      // Drop every reference to this round's raw deltas immediately -
      // both the local array and the aggregator's internal pool (via
      // advanceRound(), called here rather than after the broadcast below)
      // - so V8 can reclaim what could be hundreds of megabytes of buffers
      // before the next round starts accumulating more, without waiting on
      // the network broadcast to finish first.
      clientDeltas = [];
      aggregator.advanceRound();

      await aggregator.broadcastGlobalModel(round, globalWeights);
      console.log(`[Round ${round}] broadcast new global model (${globalWeights.length} params)`);
    }
  } finally {
    await aggregator.stop();
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const config: AggregatorServerConfig = {
    modelLength: MODEL_LENGTH, // matches modelShape.ts: inputDim x outputDim weights + outputDim bias
    clientsPerRound: process.env.NEXUS_CLIENTS_PER_ROUND ? Number(process.env.NEXUS_CLIENTS_PER_ROUND) : 2,
    roundTimeoutMs: process.env.NEXUS_ROUND_TIMEOUT_MS ? Number(process.env.NEXUS_ROUND_TIMEOUT_MS) : 15_000,
    maxRounds: process.env.NEXUS_MAX_ROUNDS ? Number(process.env.NEXUS_MAX_ROUNDS) : undefined,
    startupGraceMs: process.env.NEXUS_STARTUP_GRACE_MS ? Number(process.env.NEXUS_STARTUP_GRACE_MS) : 5_000,
    listenAddresses: ['/ip4/127.0.0.1/tcp/0/ws'],
  };
  console.log(`[AggregatorNode] model shape: inputDim=${MODEL_SHAPE.inputDim}, outputDim=${MODEL_SHAPE.outputDim}`);
  runAggregatorServer(config)
    .then(() => process.exit(0))
    .catch((err: unknown) => {
      console.error(err);
      process.exit(1);
    });
}
