import { bootstrap } from '@libp2p/bootstrap';
import { identify } from '@libp2p/identify';
import type { PubSub } from '@libp2p/interface';
import { mdns } from '@libp2p/mdns';
import { webSockets } from '@libp2p/websockets';
import * as filters from '@libp2p/websockets/filters';
import { gossipsub, type GossipsubEvents } from '@chainsafe/libp2p-gossipsub';
import { noise } from '@chainsafe/libp2p-noise';
import { yamux } from '@chainsafe/libp2p-yamux';
import { createLibp2p, type Libp2p } from 'libp2p';

/** PubSub topics shared by every node in the cohort. */
export const TOPICS = {
  GLOBAL_MODEL: 'nexus-ppfl/global-model/1.0.0',
  WEIGHT_UPDATES: 'nexus-ppfl/weight-updates/1.0.0',
  ROUND_CONTROL: 'nexus-ppfl/round-control/1.0.0',
} as const;

export type NexusTopic = (typeof TOPICS)[keyof typeof TOPICS];

export interface NexusNodeOptions {
  role: 'aggregator' | 'edge-client';
  listenAddresses?: string[];
  bootstrapPeers?: string[];
  /**
   * Auto-discover other Nexus-PPFL nodes on the same LAN segment via mDNS,
   * so edge/IoT devices don't need a pre-known aggregator address - the
   * scenario this project explicitly targets. Multicast DNS doesn't cross
   * routers/VLANs and needs UDP multicast support from the host network, so
   * it's a complement to `bootstrapPeers` (which still works across
   * networks and in restricted/sandboxed environments where multicast is
   * unavailable), not a replacement. Default true.
   */
  enableMdns?: boolean;
}

type NexusServices = {
  pubsub: PubSub<GossipsubEvents>;
};

type NexusLibp2p = Libp2p<NexusServices>;

export type NexusMessageHandler = (data: Uint8Array, fromPeerId: string | undefined) => void;

/**
 * Thin wrapper around a libp2p node configured for the Nexus-PPFL cohort:
 * WebSocket transport (browser/edge-device reachable), Noise encryption,
 * Yamux muxing, and GossipSub for topic-based broadcast of the global model,
 * client weight updates, and round-control signaling.
 */
export class NexusNetworkNode {
  private node?: NexusLibp2p;
  private readonly messageHandlers = new Map<NexusTopic, Set<NexusMessageHandler>>();

  constructor(private readonly options: NexusNodeOptions) {}

  async start(): Promise<void> {
    this.node = await createLibp2p({
      addresses: {
        listen: this.options.listenAddresses ?? ['/ip4/0.0.0.0/tcp/0/ws'],
      },
      transports: [webSockets({ filter: filters.all })],
      connectionEncrypters: [noise()],
      streamMuxers: [yamux()],
      peerDiscovery: [
        ...(this.options.bootstrapPeers?.length ? [bootstrap({ list: this.options.bootstrapPeers })] : []),
        ...(this.options.enableMdns ?? true ? [mdns()] : []),
      ],
      services: {
        identify: identify(),
        pubsub: gossipsub({
          emitSelf: false,
          allowPublishToZeroTopicPeers: true,
        }),
      },
    });

    // Neither mDNS nor bootstrap auto-dials on discovery in this libp2p
    // version - they only add the peer (and its multiaddrs) to the peer
    // store. Without this, two nodes can "discover" each other via mDNS
    // and still never open a connection, so gossipsub never meshes them.
    this.node.addEventListener('peer:discovery', (evt) => {
      const peerId = evt.detail.id;
      if (this.node?.getConnections(peerId).length) return; // already connected
      this.node?.dial(peerId).catch(() => {
        // Best-effort: the peer may be transiently unreachable, behind a
        // transport we don't share, or gone by the time we dial. Discovery
        // fires repeatedly (mDNS re-announces, bootstrap re-resolves), so a
        // failed attempt here isn't fatal - it will typically be retried.
      });
    });

    for (const topic of Object.values(TOPICS)) {
      this.node.services.pubsub.subscribe(topic);
    }

    this.node.services.pubsub.addEventListener('message', (evt) => {
      const message = evt.detail;
      const handlers = this.messageHandlers.get(message.topic as NexusTopic);
      if (!handlers || handlers.size === 0) return;
      const fromPeerId = message.type === 'signed' ? message.from.toString() : undefined;
      for (const handler of handlers) handler(message.data, fromPeerId);
    });
  }

  async stop(): Promise<void> {
    await this.node?.stop();
    this.node = undefined;
  }

  get peerId(): string {
    return this.requireNode().peerId.toString();
  }

  get multiaddrs(): string[] {
    return this.requireNode()
      .getMultiaddrs()
      .map((ma) => ma.toString());
  }

  async publish(topic: NexusTopic, payload: Uint8Array): Promise<void> {
    await this.requireNode().services.pubsub.publish(topic, payload);
  }

  /**
   * Registers a handler for `topic`. Safe to call before `start()` (message
   * delivery simply begins once the node is running) - registration and
   * dispatch are decoupled precisely so callers don't have to get this
   * ordering right themselves.
   */
  onMessage(topic: NexusTopic, handler: NexusMessageHandler): void {
    let handlers = this.messageHandlers.get(topic);
    if (!handlers) {
      handlers = new Set();
      this.messageHandlers.set(topic, handlers);
    }
    handlers.add(handler);
  }

  private requireNode(): NexusLibp2p {
    if (!this.node) {
      throw new Error('NexusNetworkNode has not been started. Call start() first.');
    }
    return this.node;
  }
}
