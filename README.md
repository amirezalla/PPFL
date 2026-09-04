# Nexus-PPFL

A Privacy-Preserving Federated Learning engine native to Node.js — edge devices train locally on `onnxruntime-node`, mask and differentially-privatize their weight updates, and exchange them over a `libp2p` gossip network with an aggregator running `FedAvg`. No client ever sends raw data or a raw gradient off the device.

Most of this space (PySyft, Flower) is Python-first. Nexus-PPFL targets the Node.js / edge / IoT / browser-to-server side of it instead.

## Status

Early-stage. The federated-averaging, differential-privacy, and secure-aggregation math is unit-tested and verified against hand-computed values (see [Testing](#testing)) — but **none of it has been reviewed by a cryptographer or independent security auditor.** Treat this as a working engineering reference, not a production-hardened privacy guarantee, until it has had that review. If you're evaluating it for real sensitive data, get an independent audit first.

## Architecture

Three packages, one monorepo:

| Package | What it does |
|---|---|
| [`@nexus-ppfl/crypto`](packages/crypto) | Differential privacy (Gaussian/Laplace mechanisms) and secure aggregation (pairwise ECDH masking + Shamir's Secret Sharing, including dropout recovery) |
| [`@nexus-ppfl/network`](packages/network) | `libp2p` PubSub node (WebSockets + GossipSub + mDNS/bootstrap discovery), Protobuf wire format, straggler-tolerant `FedAvg` coordination |
| [`@nexus-ppfl/ml-engine`](packages/ml-engine) | `EdgeTrainer` (local training via `onnxruntime-node`), `AggregatorNode` (server-side round orchestration), and the `client.ts`/`server.ts` processes that tie it all together |

A design decision worth knowing about: `onnxruntime-node` only exposes `InferenceSession`, not a training API. So `EdgeTrainer` treats an optional ONNX model as a **frozen feature extractor** and trains a small linear head on top of it with a hand-written SGD implementation — see `packages/ml-engine/src/LinearHead.ts`. If no `model.onnx` is present, the head trains directly on raw features (plain logistic regression), which is exactly what the CSV/SQLite-free demo does.

## Quick start

Requires Node.js **20+** (the multi-process cluster demo specifically wants **23.6+/24+**, since it runs a `.ts` file directly via Node's built-in TypeScript support — no `ts-node` needed).

```bash
npm install
npm run build
npm test              # 50+ unit/integration tests across all three packages
npm run cluster:test  # spins up a real aggregator + 3 edge clients as separate OS processes
```

`cluster:test` forks real child processes that talk to each other over real `libp2p` WebSocket connections on localhost — it's the closest thing here to an end-to-end demo. Watch the terminal output: each client trains, applies DP noise, and publishes a masked update; the aggregator waits for a 2-of-3 threshold, runs `FedAvg`, and broadcasts the new global model back out.

## Usage

### Differential privacy + secure aggregation (`@nexus-ppfl/crypto`)

```ts
import { PrivacyEngine, SecureAggregationClient, SecureAggregationCoordinator } from '@nexus-ppfl/crypto';

// Clip to bound sensitivity, then inject calibrated Gaussian noise.
const { noised, sigma } = PrivacyEngine.addGaussianNoise(rawDelta, {
  epsilon: 8,
  delta: 1e-5,
  sensitivity: 1,
  clipNorm: 0.3, // pick this close to your model's real gradient norm - see DEFAULT_DP_CONFIG's doc comment for why it matters
});

// Pairwise-mask a vector before it ever leaves the device.
const client = new SecureAggregationClient('device-42');
const masked = client.maskVector(noised, peerHandles);
```

### P2P networking (`@nexus-ppfl/network`)

```ts
import { NexusNetworkNode, TOPICS, WeightSerializer } from '@nexus-ppfl/network';

const node = new NexusNetworkNode({ role: 'edge-client', bootstrapPeers: [aggregatorMultiaddr] });
await node.start(); // also auto-discovers peers on the LAN via mDNS

const serializer = new WeightSerializer();
const payload = await serializer.serializeUpdate({ clientId: 'device-42', round: 0, numSamples: 128, weights: masked });
await node.publish(TOPICS.WEIGHT_UPDATES, payload);
```

### On-device training (`@nexus-ppfl/ml-engine`)

```ts
import { EdgeTrainer, writeCheckpoint } from '@nexus-ppfl/ml-engine';

// artifactDir needs checkpoint.json (baseline head params) and, optionally, model.onnx
const trainer = new EdgeTrainer();
await trainer.loadArtifacts(artifactDir);
await trainer.loadLocalData('sqlite:///device.db', { numSamples: 128 });
const report = await trainer.trainEpoch(16);
const delta = trainer.extractWeightDeltas(); // feed this into PrivacyEngine, then WeightSerializer
await trainer.dispose();
```

Full working wiring of all three packages together lives in `packages/ml-engine/src/client.ts` and `server.ts`.

## Testing

```bash
npm test
```

Runs on Node's built-in test runner (`node --test`), no test framework dependency. Covers: DP noise formulas, Shamir's Secret Sharing round-trips, secure-aggregation mask cancellation (including a dropped-client recovery scenario), Protobuf codec round-trips, `FedAvg` correctness, straggler-timeout behavior, `EdgeTrainer` convergence (with and without a real ONNX backbone — see `packages/ml-engine/fixtures/`), and a regression test that locks in correct DP calibration so noise can't silently start dominating signal again.

Test files compile to a separate `dist-test/` output and never ship in the published `dist/`.

## Known simplifications

- Every participant currently agrees on model architecture (`MODEL_SHAPE` in `packages/ml-engine/src/modelShape.ts`) out-of-band, as a shared constant, rather than exchanging it as part of a real artifact-distribution protocol.
- `FedAvg` here is an unweighted mean of client deltas (see `AggregatorNode.FedAvg`); `federatedAverage` in `@nexus-ppfl/network` is a separate, sample-weighted variant used elsewhere - they serve different aggregation strategies, not one bug.

## Contributing

Issues and PRs welcome — [github.com/amirezalla/PPFL](https://github.com/amirezalla/PPFL). Please run `npm test` before opening a PR.

## License

MIT — see [LICENSE](LICENSE).
