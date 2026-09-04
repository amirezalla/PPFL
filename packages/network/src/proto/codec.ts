import path from 'node:path';
import { fileURLToPath } from 'node:url';
import protobuf from 'protobufjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

let rootPromise: Promise<protobuf.Root> | undefined;

/** Lazily loads and memoizes the weights.proto schema on first use. */
function loadRoot(): Promise<protobuf.Root> {
  rootPromise ??= protobuf.load(path.join(__dirname, 'weights.proto'));
  return rootPromise;
}

export interface WeightUpdateMessage {
  clientId: string;
  round: number;
  numSamples: number;
  weights: Float32Array;
}

export interface GlobalModelMessage {
  round: number;
  weights: Float32Array;
  modelHash: string;
}

export type RoundControlPhase = 'UNKNOWN' | 'ROUND_START' | 'ROUND_CLOSED';

export interface RoundControlMessage {
  phase: RoundControlPhase;
  round: number;
  expectedClients: number;
  completionThreshold: number;
}

const PHASE_TO_WIRE: Record<RoundControlPhase, number> = { UNKNOWN: 0, ROUND_START: 1, ROUND_CLOSED: 2 };
const WIRE_TO_PHASE: Record<number, RoundControlPhase> = { 0: 'UNKNOWN', 1: 'ROUND_START', 2: 'ROUND_CLOSED' };

function float32ToBytes(arr: Float32Array): Uint8Array {
  return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
}

/** Copies into a freshly-allocated, 4-byte-aligned buffer before reinterpreting as Float32. */
function bytesToFloat32(bytes: Uint8Array, length: number): Float32Array {
  const aligned = new Uint8Array(bytes.byteLength);
  aligned.set(bytes);
  return new Float32Array(aligned.buffer, 0, length);
}

export async function encodeWeightUpdate(msg: WeightUpdateMessage): Promise<Uint8Array> {
  const root = await loadRoot();
  const WeightUpdate = root.lookupType('nexus.ppfl.WeightUpdate');
  const payload = {
    clientId: msg.clientId,
    round: msg.round,
    numSamples: msg.numSamples,
    weights: float32ToBytes(msg.weights),
    length: msg.weights.length,
  };
  const verifyError = WeightUpdate.verify(payload);
  if (verifyError) throw new Error(`WeightUpdate encode failed: ${verifyError}`);
  return WeightUpdate.encode(WeightUpdate.create(payload)).finish();
}

export async function decodeWeightUpdate(bytes: Uint8Array): Promise<WeightUpdateMessage> {
  const root = await loadRoot();
  const WeightUpdate = root.lookupType('nexus.ppfl.WeightUpdate');
  const decoded = WeightUpdate.decode(bytes) as unknown as {
    clientId: string;
    round: number;
    numSamples: number;
    weights: Uint8Array;
    length: number;
  };
  return {
    clientId: decoded.clientId,
    round: decoded.round,
    numSamples: decoded.numSamples,
    weights: bytesToFloat32(decoded.weights, decoded.length),
  };
}

export async function encodeGlobalModel(msg: GlobalModelMessage): Promise<Uint8Array> {
  const root = await loadRoot();
  const GlobalModel = root.lookupType('nexus.ppfl.GlobalModel');
  const payload = {
    round: msg.round,
    weights: float32ToBytes(msg.weights),
    length: msg.weights.length,
    modelHash: msg.modelHash,
  };
  const verifyError = GlobalModel.verify(payload);
  if (verifyError) throw new Error(`GlobalModel encode failed: ${verifyError}`);
  return GlobalModel.encode(GlobalModel.create(payload)).finish();
}

export async function decodeGlobalModel(bytes: Uint8Array): Promise<GlobalModelMessage> {
  const root = await loadRoot();
  const GlobalModel = root.lookupType('nexus.ppfl.GlobalModel');
  const decoded = GlobalModel.decode(bytes) as unknown as {
    round: number;
    weights: Uint8Array;
    length: number;
    modelHash: string;
  };
  return {
    round: decoded.round,
    weights: bytesToFloat32(decoded.weights, decoded.length),
    modelHash: decoded.modelHash,
  };
}

export async function encodeRoundControl(msg: RoundControlMessage): Promise<Uint8Array> {
  const root = await loadRoot();
  const RoundControl = root.lookupType('nexus.ppfl.RoundControl');
  const payload = {
    phase: PHASE_TO_WIRE[msg.phase],
    round: msg.round,
    expectedClients: msg.expectedClients,
    completionThreshold: msg.completionThreshold,
  };
  const verifyError = RoundControl.verify(payload);
  if (verifyError) throw new Error(`RoundControl encode failed: ${verifyError}`);
  return RoundControl.encode(RoundControl.create(payload)).finish();
}

export async function decodeRoundControl(bytes: Uint8Array): Promise<RoundControlMessage> {
  const root = await loadRoot();
  const RoundControl = root.lookupType('nexus.ppfl.RoundControl');
  const decoded = RoundControl.decode(bytes) as unknown as {
    phase: number;
    round: number;
    expectedClients: number;
    completionThreshold: number;
  };
  return {
    phase: WIRE_TO_PHASE[decoded.phase] ?? 'UNKNOWN',
    round: decoded.round,
    expectedClients: decoded.expectedClients,
    completionThreshold: decoded.completionThreshold,
  };
}
