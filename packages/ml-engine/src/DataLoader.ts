import { readFile } from 'node:fs/promises';
import type { DataSource, LocalDataset } from './types.js';

export interface CsvDataSourceOptions {
  featureColumns: string[];
  labelColumn: string;
}

/**
 * Minimal CSV loader for local, non-IID edge data - the file never leaves
 * the device. Returns one flattened `LocalDataset`; batching happens later,
 * inside `EdgeTrainer.trainEpoch`, so this stays a pure I/O + parsing step.
 */
export class CsvDataSource implements DataSource {
  constructor(
    private readonly filePath: string,
    private readonly options: CsvDataSourceOptions,
  ) {}

  async load(): Promise<LocalDataset> {
    const raw = await readFile(this.filePath, 'utf8');
    const lines = raw.trim().split('\n');
    const headerLine = lines[0];
    if (!headerLine) throw new Error(`${this.filePath} is empty.`);

    const headers = headerLine.split(',').map((h) => h.trim());
    const featureIdx = this.options.featureColumns.map((c) => headers.indexOf(c));
    const labelIdx = headers.indexOf(this.options.labelColumn);
    if (featureIdx.includes(-1) || labelIdx === -1) {
      throw new Error(`${this.filePath} header is missing one or more configured columns.`);
    }

    const rows = lines
      .slice(1)
      .filter((line) => line.length > 0)
      .map((line) => line.split(','));

    const inputDim = featureIdx.length;
    const features = new Float32Array(rows.length * inputDim);
    const labels = new Float32Array(rows.length);

    rows.forEach((row, rowIdx) => {
      featureIdx.forEach((colIdx, featIdx) => {
        features[rowIdx * inputDim + featIdx] = parseFloat(row[colIdx] ?? '0');
      });
      labels[rowIdx] = parseFloat(row[labelIdx] ?? '0');
    });

    return { features, labels, numSamples: rows.length, inputDim };
  }
}
