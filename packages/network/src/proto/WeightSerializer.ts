import {
  decodeGlobalModel,
  decodeRoundControl,
  decodeWeightUpdate,
  encodeGlobalModel,
  encodeRoundControl,
  encodeWeightUpdate,
  type GlobalModelMessage,
  type RoundControlMessage,
  type WeightUpdateMessage,
} from './codec.js';

/**
 * Class-based facade over weights.proto's codec functions. Nexus-PPFL's
 * wire format for "a client's weight update" is the `WeightUpdate` message
 * (clientId, round, weights-as-bytes, numSamples) - this wraps it rather
 * than introducing a second, competing schema for the same concept.
 */
export class WeightSerializer {
  serializeUpdate(message: WeightUpdateMessage): Promise<Uint8Array> {
    return encodeWeightUpdate(message);
  }

  deserializeUpdate(bytes: Uint8Array): Promise<WeightUpdateMessage> {
    return decodeWeightUpdate(bytes);
  }

  serializeGlobalModel(message: GlobalModelMessage): Promise<Uint8Array> {
    return encodeGlobalModel(message);
  }

  deserializeGlobalModel(bytes: Uint8Array): Promise<GlobalModelMessage> {
    return decodeGlobalModel(bytes);
  }

  serializeRoundControl(message: RoundControlMessage): Promise<Uint8Array> {
    return encodeRoundControl(message);
  }

  deserializeRoundControl(bytes: Uint8Array): Promise<RoundControlMessage> {
    return decodeRoundControl(bytes);
  }
}
