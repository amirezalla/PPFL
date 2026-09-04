/**
 * onnxruntime-node exposes inference only (no on-device TrainingSession, unlike
 * onnxruntime-web) - so edge clients can't backprop through an arbitrary ONNX
 * graph. Instead the ONNX model is treated as a frozen feature extractor, and
 * this linear head (logits = W*x + b) is the one piece each client actually
 * trains locally: manual softmax cross-entropy backprop and plain SGD, small
 * enough to keep the produced weight-delta (and thus the noise/mask math in
 * the crypto layer) simple and bounded.
 */
export interface HeadParameters {
  inputDim: number;
  outputDim: number;
  /** Row-major, outputDim x inputDim. */
  weights: Float32Array;
  bias: Float32Array;
}

export function cloneHeadParameters(head: HeadParameters): HeadParameters {
  return {
    inputDim: head.inputDim,
    outputDim: head.outputDim,
    weights: new Float32Array(head.weights),
    bias: new Float32Array(head.bias),
  };
}

export function flattenHeadParameters(head: HeadParameters): Float32Array {
  const flat = new Float32Array(head.weights.length + head.bias.length);
  flat.set(head.weights, 0);
  flat.set(head.bias, head.weights.length);
  return flat;
}

export function unflattenHeadParameters(
  flat: Float32Array,
  inputDim: number,
  outputDim: number,
): HeadParameters {
  const weightsLength = inputDim * outputDim;
  return {
    inputDim,
    outputDim,
    weights: flat.slice(0, weightsLength),
    bias: flat.slice(weightsLength, weightsLength + outputDim),
  };
}

export function subtractFlat(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i] ?? 0) - (b[i] ?? 0);
  return out;
}

function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp((logits[i] ?? 0) - max);
    exps[i] = e;
    sum += e;
  }
  for (let i = 0; i < exps.length; i++) exps[i] = (exps[i] ?? 0) / sum;
  return exps;
}

export class LinearHead {
  constructor(private params: HeadParameters) {}

  getParameters(): HeadParameters {
    return this.params;
  }

  /**
   * One SGD step over a batch of embeddings, mutating this head's parameters
   * in place. `labels` holds integer class indices as floats. Returns the
   * batch's mean cross-entropy loss.
   */
  trainStep(embeddings: Float32Array, batchSize: number, labels: Float32Array, learningRate: number): number {
    const { inputDim, outputDim, weights, bias } = this.params;
    const gradWeights = new Float32Array(weights.length);
    const gradBias = new Float32Array(bias.length);
    let totalLoss = 0;

    for (let n = 0; n < batchSize; n++) {
      const x = embeddings.subarray(n * inputDim, (n + 1) * inputDim);
      const logits = new Float32Array(outputDim);
      for (let o = 0; o < outputDim; o++) {
        let sum = bias[o] ?? 0;
        for (let i = 0; i < inputDim; i++) {
          sum += (weights[o * inputDim + i] ?? 0) * (x[i] ?? 0);
        }
        logits[o] = sum;
      }

      const probs = softmax(logits);
      const label = Math.round(labels[n] ?? 0);
      totalLoss += -Math.log(Math.max(probs[label] ?? 1e-12, 1e-12));

      for (let o = 0; o < outputDim; o++) {
        const grad = (probs[o] ?? 0) - (o === label ? 1 : 0);
        gradBias[o] = (gradBias[o] ?? 0) + grad;
        for (let i = 0; i < inputDim; i++) {
          const idx = o * inputDim + i;
          gradWeights[idx] = (gradWeights[idx] ?? 0) + grad * (x[i] ?? 0);
        }
      }
    }

    for (let idx = 0; idx < weights.length; idx++) {
      weights[idx] = (weights[idx] ?? 0) - (learningRate * (gradWeights[idx] ?? 0)) / batchSize;
    }
    for (let o = 0; o < outputDim; o++) {
      bias[o] = (bias[o] ?? 0) - (learningRate * (gradBias[o] ?? 0)) / batchSize;
    }

    return totalLoss / batchSize;
  }
}
