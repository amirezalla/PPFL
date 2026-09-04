import { createHash } from 'node:crypto';
import { NexusNetworkNode, TOPICS, WeightSerializer } from '@nexus-ppfl/network';

export interface AggregatorNodeOptions {
  listenAddresses?: string[];
}

interface PendingWait {
  k: number;
  settled: boolean;
  resolve: () => void;
}

/**
 * Server-mode wrapper around Phase 1's `NexusNetworkNode`: listens for
 * client `WeightUpdate` payloads on the PubSub weight-updates topic,
 * accumulates them per round, and exposes the straggler-tolerant
 * `waitForThreshold` primitive the aggregation loop (see server.ts) drives.
 */
export class AggregatorNode {
  private readonly node: NexusNetworkNode;
  private readonly serializer = new WeightSerializer();
  private currentRoundValue = 0;
  private readonly receivedUpdates = new Map<string, Float32Array>();
  private readonly pendingWaits: PendingWait[] = [];

  constructor(options: AggregatorNodeOptions = {}) {
    this.node = new NexusNetworkNode({ role: 'aggregator', listenAddresses: options.listenAddresses });
  }

  get round(): number {
    return this.currentRoundValue;
  }

  get peerId(): string {
    return this.node.peerId;
  }

  get multiaddrs(): string[] {
    return this.node.multiaddrs;
  }

  get receivedCount(): number {
    return this.receivedUpdates.size;
  }

  async start(): Promise<void> {
    await this.node.start();
    this.node.onMessage(TOPICS.WEIGHT_UPDATES, (data) => {
      void this.handleIncomingUpdate(data);
    });
  }

  async stop(): Promise<void> {
    await this.node.stop();
  }

  /**
   * Publishes a RoundControl "epoch started" message so edge clients know
   * to begin training. `round` is taken explicitly (rather than read from
   * internal state at call time) so callers control exactly which round a
   * broadcast is tagged with, independent of when `advanceRound()` runs.
   */
  async broadcastRoundStart(round: number, expectedClients: number, completionThreshold = 1): Promise<void> {
    const payload = await this.serializer.serializeRoundControl({
      phase: 'ROUND_START',
      round,
      expectedClients,
      completionThreshold,
    });
    await this.node.publish(TOPICS.ROUND_CONTROL, payload);
  }

  /** Publishes the new global model checkpoint, tagged with the round that produced it. */
  async broadcastGlobalModel(round: number, weights: Float32Array): Promise<void> {
    const modelHash = createHash('sha256')
      .update(Buffer.from(weights.buffer, weights.byteOffset, weights.byteLength))
      .digest('hex');
    const payload = await this.serializer.serializeGlobalModel({ round, weights, modelHash });
    await this.node.publish(TOPICS.GLOBAL_MODEL, payload);
  }

  /**
   * Resolves once `k` distinct clients have reported an update for the
   * current round, or `timeoutMs` elapses - whichever comes first. Never
   * rejects: a round that times out below threshold still resolves with
   * whatever arrived, which is what makes this straggler-tolerant (the
   * caller decides whether "fewer than k" is still enough to run FedAvg;
   * see the equivalent completion-fraction design in Phase 1's
   * `FederatedAggregationRound`).
   */
  async waitForThreshold(k: number, timeoutMs: number): Promise<Float32Array[]> {
    if (this.receivedUpdates.size < k) {
      await new Promise<void>((resolve) => {
        const wait: PendingWait = { k, settled: false, resolve };
        this.pendingWaits.push(wait);
        setTimeout(() => this.settleWait(wait), timeoutMs);
      });
    }
    return [...this.receivedUpdates.values()];
  }

  /** Drops this round's update pool - freeing every buffered Float32Array for GC - and advances the round. */
  advanceRound(): void {
    this.receivedUpdates.clear();
    this.currentRoundValue += 1;
  }

  private async handleIncomingUpdate(data: Uint8Array): Promise<void> {
    const message = await this.serializer.deserializeUpdate(data);
    if (message.round !== this.currentRoundValue) return; // stale or future round; ignore
    if (this.receivedUpdates.has(message.clientId)) return; // dedup a redelivered/duplicate submission
    this.receivedUpdates.set(message.clientId, message.weights);
    this.checkWaiters();
  }

  private checkWaiters(): void {
    for (const wait of this.pendingWaits) {
      if (!wait.settled && this.receivedUpdates.size >= wait.k) {
        this.settleWait(wait);
      }
    }
  }

  private settleWait(wait: PendingWait): void {
    if (wait.settled) return;
    wait.settled = true;
    const index = this.pendingWaits.indexOf(wait);
    if (index !== -1) this.pendingWaits.splice(index, 1);
    wait.resolve();
  }

  /**
   * Averages every client's local weight delta and applies the result to
   * the current global weights: `globalWeights[i] += mean(clientDeltas[*][i])`.
   * Returns a new array; `globalWeights` is not mutated.
   *
   * SECURE AGGREGATION MASK CANCELLATION (Phase 1, @nexus-ppfl/crypto): each
   * `clientDeltas[c]` here is exactly the payload
   * `SecureAggregationClient.maskVector()` produces - the client's real
   * delta plus a signed pairwise mask per peer, plus a private self-mask.
   * The pairwise masks are constructed with opposite signs for the two
   * clients in a pair (+1 for the lexicographically smaller clientId, -1
   * for the other), both expanded from the same ECDH-derived seed - so
   * summing every pair's contribution in the accumulation loop below
   * cancels them exactly: for clients u < v, u contributes +PRG(s_uv) and v
   * contributes -PRG(s_uv) at the same index, netting zero once both are in
   * the sum. What survives this loop is the true sum of deltas plus every
   * contributor's uncancelled self-mask; a
   * `SecureAggregationCoordinator.removeSelfMasks` pass (fed the
   * threshold-reconstructed self-mask seeds gathered via each client's
   * `shareSeed`) must run on the deltas - or on the summed result - before
   * genuinely masked input reaches this method, or the self-mask noise
   * floor shows up as extra variance in the resulting global weights. This
   * method itself is mask-agnostic: it only sums and averages whatever
   * Float32Arrays it is given.
   */
  static FedAvg(globalWeights: Float32Array, clientDeltas: Float32Array[]): Float32Array {
    const length = globalWeights.length;
    for (const delta of clientDeltas) {
      if (delta.length !== length) {
        throw new Error(`Weight delta length ${delta.length} does not match global model length ${length}.`);
      }
    }

    const updated = new Float32Array(globalWeights);
    const numClients = clientDeltas.length;
    if (numClients === 0) return updated;

    // Client-outer / index-inner accumulation: each clientDeltas[c] is
    // walked exactly once, sequentially (a single linear TypedArray scan,
    // no allocations), accumulating straight into `updated`. This is the
    // hot path for models with millions of parameters, so there is
    // deliberately no .map()/.reduce() here - just direct indexed writes
    // that V8 JITs to near-native code. The only buffer beyond the inputs
    // is `updated` itself, sized once up front.
    const inverseCount = 1 / numClients;
    for (let c = 0; c < numClients; c++) {
      const delta = clientDeltas[c]!;
      for (let i = 0; i < length; i++) {
        updated[i] = (updated[i] ?? 0) + (delta[i] ?? 0) * inverseCount;
      }
    }

    return updated;
  }
}
