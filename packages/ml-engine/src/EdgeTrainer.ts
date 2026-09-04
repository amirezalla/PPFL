import { access, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import * as ort from 'onnxruntime-node';
import {
  LinearHead,
  cloneHeadParameters,
  flattenHeadParameters,
  subtractFlat,
  type HeadParameters,
} from './LinearHead.js';
import type { DataSource, EpochReport, LocalDataset } from './types.js';

/**
 * SCOPE NOTE: the published `onnxruntime-node` binding exposes
 * `InferenceSession` only - there is no `TrainingSession` in its API surface
 * (nor, as of the currently published `onnxruntime-web`, in its public
 * types either). Backprop through an arbitrary ONNX graph isn't available
 * on Node today, so EdgeTrainer trains the way it actually can:
 *  - an optional frozen ONNX model (`model.onnx`) is run through
 *    `InferenceSession` as a feature extractor - this is the real
 *    onnxruntime-node usage, and
 *  - a small linear classification head on top of its output is trained
 *    with a from-scratch SGD implementation (`LinearHead`, manual softmax
 *    cross-entropy backprop).
 * When `model.onnx` isn't present, the head trains directly on the raw
 * input vector - i.e. plain multinomial logistic regression - which suits
 * lightweight edge/IoT devices with no backbone at all.
 *
 * `checkpoint.json` stands in for the checkpoint/optimizer-state artifact
 * ORT's native (Python/C++) training stack would otherwise provide, holding
 * this round's baseline head parameters as `{ inputDim, outputDim, weights,
 * bias }`.
 */

export type EdgeCheckpoint = HeadParameters;

export async function readCheckpoint(filePath: string): Promise<EdgeCheckpoint> {
  const raw = await readFile(filePath, 'utf8');
  const parsed = JSON.parse(raw) as {
    inputDim: number;
    outputDim: number;
    weights: number[];
    bias: number[];
  };
  return {
    inputDim: parsed.inputDim,
    outputDim: parsed.outputDim,
    weights: Float32Array.from(parsed.weights),
    bias: Float32Array.from(parsed.bias),
  };
}

export async function writeCheckpoint(filePath: string, checkpoint: EdgeCheckpoint): Promise<void> {
  await writeFile(
    filePath,
    JSON.stringify({
      inputDim: checkpoint.inputDim,
      outputDim: checkpoint.outputDim,
      weights: Array.from(checkpoint.weights),
      bias: Array.from(checkpoint.bias),
    }),
  );
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

/** Deterministic string -> 32-bit seed, so each simulated client gets a stable, distinct RNG stream. */
function hashSeed(input: string): number {
  let h = 2166136261;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class EdgeTrainer {
  private backbone?: ort.InferenceSession;
  private head?: LinearHead;
  private baseline?: EdgeCheckpoint;
  private dataset?: LocalDataset;

  /**
   * Loads this round's artifacts from `artifactDir`:
   *  - `checkpoint.json` (required): baseline head parameters for this round.
   *  - `model.onnx` (optional): a frozen ONNX feature extractor. If absent,
   *    the head trains directly on raw input features.
   */
  async loadArtifacts(artifactDir: string): Promise<void> {
    const modelPath = path.join(artifactDir, 'model.onnx');
    const checkpointPath = path.join(artifactDir, 'checkpoint.json');

    this.baseline = await readCheckpoint(checkpointPath);
    this.head = new LinearHead(cloneHeadParameters(this.baseline));
    this.backbone = (await fileExists(modelPath)) ? await ort.InferenceSession.create(modelPath) : undefined;
  }

  /**
   * Mock local-data loader: no real device store is read here. In
   * production this would pull from a per-device SQLite DB or CSV (see
   * `CsvDataSource` / `loadDataFrom`) and the data would never leave the
   * device; here it synthesizes a small non-IID classification set so the
   * training pipeline is runnable end-to-end without real user data.
   * `dbPath` seeds the RNG purely so different simulated clients land on
   * systematically different class distributions, standing in for genuine
   * non-IID skew across edge devices.
   */
  async loadLocalData(dbPath: string, options: { numSamples?: number } = {}): Promise<LocalDataset> {
    const { inputDim, outputDim } = this.requireBaseline();
    const numSamples = options.numSamples ?? 64;
    const rng = mulberry32(hashSeed(dbPath));
    const dominantClass = Math.floor(rng() * outputDim);

    const features = new Float32Array(numSamples * inputDim);
    const labels = new Float32Array(numSamples);
    for (let n = 0; n < numSamples; n++) {
      // 70% of samples land on this simulated device's dominant class - the non-IID skew.
      const label = rng() < 0.7 ? dominantClass : Math.floor(rng() * outputDim);
      labels[n] = label;
      for (let i = 0; i < inputDim; i++) {
        features[n * inputDim + i] = (rng() - 0.5) * 2 + (i === label % inputDim ? 1.5 : 0);
      }
    }

    const dataset: LocalDataset = { features, labels, numSamples, inputDim };
    this.dataset = dataset;
    return dataset;
  }

  /** Loads real local data through a pluggable source (e.g. `CsvDataSource`) instead of the mock generator. */
  async loadDataFrom(source: DataSource): Promise<LocalDataset> {
    this.dataset = await source.load();
    return this.dataset;
  }

  /**
   * Slices the loaded dataset into batches of `batchSize` and runs one pass
   * over all of them. Each batch: if a backbone is loaded, runs it through
   * `InferenceSession.run` to get embeddings (shape [batchSize, inputDim]
   * in, [batchSize, embeddingDim] out); otherwise the raw feature slice
   * itself is the "embedding". Either way, one manual SGD step is taken on
   * the linear head. Batches are `subarray` views (no copy); the only heap
   * growth per batch is the small gradient buffer scoped inside
   * `LinearHead.trainStep`, which is discarded each iteration - no tensors
   * or buffers are retained across batches.
   */
  async trainEpoch(batchSize: number, learningRate = 0.05): Promise<EpochReport> {
    const head = this.requireHead();
    const dataset = this.requireDataset();

    let totalLoss = 0;
    let batches = 0;

    for (let start = 0; start < dataset.numSamples; start += batchSize) {
      const end = Math.min(start + batchSize, dataset.numSamples);
      const currentBatchSize = end - start;
      const batchLabels = dataset.labels.subarray(start, end);
      const batchFeatures = dataset.features.subarray(start * dataset.inputDim, end * dataset.inputDim);

      const embeddings = this.backbone
        ? await this.runBackbone(batchFeatures, currentBatchSize, dataset.inputDim)
        : batchFeatures;

      totalLoss += head.trainStep(embeddings, currentBatchSize, batchLabels, learningRate);
      batches += 1;
    }

    return { numSamples: dataset.numSamples, batches, avgLoss: totalLoss / Math.max(batches, 1) };
  }

  /** Flattened difference between the trained head and this round's baseline checkpoint - the weight delta. */
  extractWeightDeltas(): Float32Array {
    return subtractFlat(
      flattenHeadParameters(this.requireHead().getParameters()),
      flattenHeadParameters(this.requireBaseline()),
    );
  }

  /** Releases the native inference session. Call once this round's training is done. */
  async dispose(): Promise<void> {
    await this.backbone?.release();
    this.backbone = undefined;
    this.head = undefined;
    this.dataset = undefined;
  }

  private async runBackbone(
    batchFeatures: Float32Array,
    batchSize: number,
    inputDim: number,
  ): Promise<Float32Array> {
    const backbone = this.backbone;
    if (!backbone) throw new Error('No backbone loaded.');
    const inputName = backbone.inputNames[0];
    const outputName = backbone.outputNames[0];
    if (!inputName || !outputName) throw new Error('Backbone model exposes no inputs/outputs.');

    const inputTensor = new ort.Tensor('float32', batchFeatures, [batchSize, inputDim]);
    const results = await backbone.run({ [inputName]: inputTensor });
    const embeddingTensor = results[outputName];
    if (!embeddingTensor) throw new Error(`Backbone output "${outputName}" not found.`);
    return embeddingTensor.data as Float32Array;
  }

  private requireHead(): LinearHead {
    if (!this.head) throw new Error('Call loadArtifacts() before training.');
    return this.head;
  }

  private requireBaseline(): EdgeCheckpoint {
    if (!this.baseline) throw new Error('Call loadArtifacts() before training.');
    return this.baseline;
  }

  private requireDataset(): LocalDataset {
    if (!this.dataset) throw new Error('Call loadLocalData() or loadDataFrom() before training.');
    return this.dataset;
  }
}
