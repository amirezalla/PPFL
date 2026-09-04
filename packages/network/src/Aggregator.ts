/**
 * FedAvg aggregation for a single training round, with straggler handling:
 * the aggregator does not wait for every expected client. It settles as soon
 * as a configurable completion threshold (e.g. 80% of 10,000 nodes) has
 * reported in, or a hard timeout elapses - whichever comes first - and
 * proceeds with whoever showed up.
 */

export interface ClientUpdate {
  clientId: string;
  weights: Float32Array;
  numSamples: number;
}

export interface AggregationRoundOptions {
  /** Total number of clients invited into this round. */
  expectedClients: number;
  /** Fraction of expectedClients required before settling early, e.g. 0.8. Default 0.8. */
  completionThreshold?: number;
  /** Hard ceiling on how long to wait for stragglers, in ms. */
  roundTimeoutMs?: number;
  /** Absolute floor on participants, regardless of the threshold fraction. Default 1. */
  minClients?: number;
}

export interface AggregationResult {
  weights: Float32Array;
  participantCount: number;
  droppedStragglers: string[];
}

/** Sample-size-weighted average of per-client weight vectors (standard FedAvg). */
export function federatedAverage(updates: ClientUpdate[]): Float32Array {
  if (updates.length === 0) {
    throw new Error('Cannot aggregate an empty set of updates.');
  }
  const length = updates[0]!.weights.length;
  const totalSamples = updates.reduce((sum, u) => sum + u.numSamples, 0);
  if (totalSamples === 0) {
    throw new Error('Cannot aggregate updates that report zero total samples.');
  }

  const result = new Float32Array(length);
  for (const update of updates) {
    if (update.weights.length !== length) {
      throw new Error(`Weight length mismatch for client ${update.clientId}`);
    }
    const weight = update.numSamples / totalSamples;
    for (let i = 0; i < length; i++) {
      result[i] = (result[i] ?? 0) + (update.weights[i] ?? 0) * weight;
    }
  }
  return result;
}

/**
 * Coordinates one round of asynchronous client submissions. Call
 * `registerExpectedClient` for each invited client, then `start()` to begin
 * the timeout clock, and feed submissions in via `submitUpdate` as they
 * arrive over the network. The returned promise resolves the moment the
 * completion threshold is met or the timeout fires, whichever is first.
 */
export class FederatedAggregationRound {
  private readonly updates = new Map<string, ClientUpdate>();
  private readonly expectedClientIds = new Set<string>();
  private readonly resultPromise: Promise<AggregationResult>;
  private resolveRound?: (result: AggregationResult) => void;
  private timeoutHandle?: ReturnType<typeof setTimeout>;
  private settled = false;

  constructor(private readonly options: AggregationRoundOptions) {
    this.resultPromise = new Promise((resolve) => {
      this.resolveRound = resolve;
    });
  }

  registerExpectedClient(clientId: string): void {
    this.expectedClientIds.add(clientId);
  }

  submitUpdate(update: ClientUpdate): void {
    if (this.settled) return;
    this.updates.set(update.clientId, update);

    const threshold = this.options.completionThreshold ?? 0.8;
    const required = Math.max(
      this.options.minClients ?? 1,
      Math.ceil(this.options.expectedClients * threshold),
    );
    if (this.updates.size >= required) {
      this.settle();
    }
  }

  /** Begins the straggler timeout (if configured) and returns the round's eventual result. */
  start(): Promise<AggregationResult> {
    if (this.options.roundTimeoutMs !== undefined) {
      this.timeoutHandle = setTimeout(() => this.settle(), this.options.roundTimeoutMs);
    }
    return this.resultPromise;
  }

  private settle(): void {
    if (this.settled) return;
    this.settled = true;
    if (this.timeoutHandle) clearTimeout(this.timeoutHandle);

    const participants = [...this.updates.values()];
    const droppedStragglers = [...this.expectedClientIds].filter((id) => !this.updates.has(id));

    this.resolveRound?.({
      weights: federatedAverage(participants),
      participantCount: participants.length,
      droppedStragglers,
    });
  }
}
