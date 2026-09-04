export {
  NexusNetworkNode,
  TOPICS,
  type NexusNodeOptions,
  type NexusTopic,
  type NexusMessageHandler,
} from './NexusNetworkNode.js';

export {
  FederatedAggregationRound,
  federatedAverage,
  type ClientUpdate,
  type AggregationRoundOptions,
  type AggregationResult,
} from './Aggregator.js';

export {
  encodeWeightUpdate,
  decodeWeightUpdate,
  encodeGlobalModel,
  decodeGlobalModel,
  encodeRoundControl,
  decodeRoundControl,
  type WeightUpdateMessage,
  type GlobalModelMessage,
  type RoundControlMessage,
  type RoundControlPhase,
} from './proto/codec.js';

export { WeightSerializer } from './proto/WeightSerializer.js';
