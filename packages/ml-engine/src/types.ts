/**
 * A single client's local data, flattened for zero-copy batch slicing.
 * Shape assumption: a simple multinomial classification problem - `features`
 * is row-major [numSamples, inputDim] and `labels` is [numSamples] integer
 * class indices in [0, outputDim), stored as float32 for tensor-friendliness.
 */
export interface LocalDataset {
  features: Float32Array;
  labels: Float32Array;
  numSamples: number;
  inputDim: number;
}

/** Pluggable local data provider - implement this for a real on-device store (SQLite, IndexedDB, ...). */
export interface DataSource {
  load(): Promise<LocalDataset>;
}

export interface EpochReport {
  numSamples: number;
  batches: number;
  avgLoss: number;
}
