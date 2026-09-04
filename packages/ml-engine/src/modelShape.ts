/**
 * Shared model architecture, agreed out-of-band by every participant before
 * a training run starts. In production this would be baked into the
 * distributed model.onnx/checkpoint artifact metadata rather than
 * hardcoded; the local cluster test harness has no such artifact-exchange
 * channel, so both the aggregator (server.ts) and every edge client
 * (client.ts) import this constant directly instead.
 */
export const MODEL_SHAPE = { inputDim: 8, outputDim: 3 } as const;

export const MODEL_LENGTH = MODEL_SHAPE.inputDim * MODEL_SHAPE.outputDim + MODEL_SHAPE.outputDim;

/**
 * Default differential-privacy budget for the demo and cluster-test
 * clients, empirically calibrated against this model's actual per-round
 * gradient L2 norm (~0.24 on the synthetic dataset EdgeTrainer.loadLocalData
 * generates). Earlier defaults here (epsilon=2, clipNorm=1.0) produced a
 * Gaussian noise sigma about 20x larger than the entire real delta
 * vector's norm - every round's aggregate was pure noise, so the global
 * model diverged instead of converging (observed directly: client loss
 * climbing from ~1.0 to ~11 after a single FedAvg round in the multi-process
 * cluster test). These values keep injected noise on the same order of
 * magnitude as the real signal instead. A real deployment must recalibrate
 * clipNorm/epsilon against its own model's gradient-norm profile and threat
 * model - these numbers are specific to this toy linear head, not universal.
 */
export const DEFAULT_DP_CONFIG = {
  epsilon: 10,
  delta: 1e-5,
  sensitivity: 1,
  clipNorm: 0.3,
} as const;
