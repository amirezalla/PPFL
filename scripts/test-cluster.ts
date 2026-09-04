/**
 * Local multi-process integration harness: forks one AggregatorNode and
 * three edge clients from their built dist/ output, wires the aggregator's
 * multiaddr through to the clients over the fork() IPC channel (no fixed
 * port or file polling needed), and lets a full federated round run over
 * real libp2p connections between separate OS processes.
 *
 * Run with `npm run cluster:test` from the repo root after `npm run build`.
 * Node 23.6+/24+ runs this file directly via built-in TypeScript type
 * stripping - no ts-node required.
 */
import { fork, type ChildProcess } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const AGGREGATOR_ENTRY = path.join(__dirname, '../packages/ml-engine/dist/server.js');
const CLIENT_ENTRY = path.join(__dirname, '../packages/ml-engine/dist/client.js');
const CLIENT_IDS = ['client_1', 'client_2', 'client_3'];

interface AggregatorReadyMessage {
  type: 'aggregator-ready';
  peerId: string;
  multiaddrs: string[];
}

function isAggregatorReady(message: unknown): message is AggregatorReadyMessage {
  return (
    typeof message === 'object' &&
    message !== null &&
    (message as { type?: unknown }).type === 'aggregator-ready' &&
    Array.isArray((message as { multiaddrs?: unknown }).multiaddrs)
  );
}

const children: ChildProcess[] = [];
let shuttingDown = false;

function killAll(): void {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children) {
    if (!child.killed) child.kill();
  }
}

process.on('SIGINT', () => {
  console.log('\nInterrupted - stopping cluster.');
  killAll();
  process.exit(0);
});

console.log('Starting Aggregator Node...');
const aggregator = fork(AGGREGATOR_ENTRY, [], {
  env: {
    ...process.env,
    NEXUS_CLIENTS_PER_ROUND: '2', // 2-of-3 threshold, so the cluster tolerates one straggler
    NEXUS_ROUND_TIMEOUT_MS: '8000',
    NEXUS_MAX_ROUNDS: '3',
  },
});
children.push(aggregator);

aggregator.on('exit', (code) => {
  console.log(`Aggregator exited with code ${code}; stopping remaining clients.`);
  killAll();
  process.exitCode = code ?? 0;
});

aggregator.on('message', (message: unknown) => {
  if (!isAggregatorReady(message)) return;

  const wsAddr = message.multiaddrs.find((addr) => addr.includes('/ws'));
  if (!wsAddr) {
    console.error('Aggregator did not report a WebSocket multiaddr; aborting.');
    killAll();
    process.exit(1);
  }

  console.log(`Aggregator ready at ${wsAddr}`);
  console.log(`Launching ${CLIENT_IDS.length} edge client nodes...`);
  for (const id of CLIENT_IDS) {
    const client = fork(CLIENT_ENTRY, [`--id=${id}`, `--aggregator=${wsAddr}`]);
    children.push(client);
    client.on('exit', (code) => {
      console.log(`[${id}] exited with code ${code}`);
    });
  }
});
