import _ from "lodash";
import { BaseNode } from "../../Node";
import { defaultFS, path } from "../../env";
import { type GenericFileSystem, exists } from "../FileSystem";
import { DEFAULT_PERSIST_DIR } from "../constants";
import {
  type VectorStore,
  type VectorStoreQuery,
  VectorStoreQueryMode,
  type VectorStoreQueryResult,
} from "./types";
import { DEFAULT_SIMILARITY_TOP_K } from '../../constants'

/**
 * Similarity type
 * Default is cosine similarity. Dot product and negative Euclidean distance are also supported.
 */
export enum SimilarityType {
  DEFAULT = "cosine",
  DOT_PRODUCT = "dot_product",
  EUCLIDEAN = "euclidean",
}

/**
 * The similarity between two embeddings.
 * @param embedding1
 * @param embedding2
 * @param mode
 * @returns similarity score with higher numbers meaning the two embeddings are more similar
 */

export function similarity(
  embedding1: number[],
  embedding2: number[],
  mode: SimilarityType = SimilarityType.DEFAULT,
): number {
  if (embedding1.length !== embedding2.length) {
    throw new Error("Embedding length mismatch");
  }

  // NOTE I've taken enough Kahan to know that we should probably leave the
  // numeric programming to numeric programmers. The naive approach here
  // will probably cause some avoidable loss of floating point precision
  // ml-distance is worth watching although they currently also use the naive
  // formulas
  function norm(x: number[]): number {
    let result = 0;
    for (let i = 0; i < x.length; i++) {
      result += x[i] * x[i];
    }
    return Math.sqrt(result);
  }

  switch (mode) {
    case SimilarityType.EUCLIDEAN: {
      let difference = embedding1.map((x, i) => x - embedding2[i]);
      return -norm(difference);
    }
    case SimilarityType.DOT_PRODUCT: {
      let result = 0;
      for (let i = 0; i < embedding1.length; i++) {
        result += embedding1[i] * embedding2[i];
      }
      return result;
    }
    case SimilarityType.DEFAULT: {
      return (
        similarity(embedding1, embedding2, SimilarityType.DOT_PRODUCT) /
        (norm(embedding1) * norm(embedding2))
      );
    }
    default:
      throw new Error("Not implemented yet");
  }
}

/**
 * Get the top K embeddings from a list of embeddings ordered by similarity to the query.
 * @param queryEmbedding
 * @param embeddings list of embeddings to consider
 * @param similarityTopK max number of embeddings to return, default 2
 * @param embeddingIds ids of embeddings in the embeddings list
 * @param similarityCutoff minimum similarity score
 * @returns
 */
// eslint-disable-next-line max-params
export function getTopKEmbeddings(
  queryEmbedding: number[],
  embeddings: number[][],
  similarityTopK: number = DEFAULT_SIMILARITY_TOP_K,
  embeddingIds: any[] | null = null,
  similarityCutoff: number | null = null,
): [number[], any[]] {
  if (embeddingIds == null) {
    embeddingIds = Array(embeddings.length).map((_, i) => i);
  }

  if (embeddingIds.length !== embeddings.length) {
    throw new Error(
      "getTopKEmbeddings: embeddings and embeddingIds length mismatch",
    );
  }

  let similarities: { similarity: number; id: number }[] = [];

  for (let i = 0; i < embeddings.length; i++) {
    const sim = similarity(queryEmbedding, embeddings[i]);
    if (similarityCutoff == null || sim > similarityCutoff) {
      similarities.push({ similarity: sim, id: embeddingIds[i] });
    }
  }

  similarities.sort((a, b) => b.similarity - a.similarity); // Reverse sort

  let resultSimilarities: number[] = [];
  let resultIds: any[] = [];

  for (let i = 0; i < similarityTopK; i++) {
    if (i >= similarities.length) {
      break;
    }
    resultSimilarities.push(similarities[i].similarity);
    resultIds.push(similarities[i].id);
  }

  return [resultSimilarities, resultIds];
}

// eslint-disable-next-line max-params
export function getTopKEmbeddingsLearner(
  queryEmbedding: number[],
  embeddings: number[][],
  similarityTopK?: number,
  embeddingsIds?: any[],
  queryMode: VectorStoreQueryMode = VectorStoreQueryMode.SVM,
): [number[], any[]] {
  throw new Error("Not implemented yet");
  // To support SVM properly we're probably going to have to use something like
  // https://github.com/mljs/libsvm which itself hasn't been updated in a while
}

// eslint-disable-next-line max-params
export function getTopKMMREmbeddings(
  queryEmbedding: number[],
  embeddings: number[][],
  similarityFn: ((...args: any[]) => number) | null = null,
  similarityTopK: number | null = null,
  embeddingIds: any[] | null = null,
  _similarityCutoff: number | null = null,
  mmrThreshold: number | null = null,
): [number[], any[]] {
  let threshold = mmrThreshold || 0.5;
  similarityFn = similarityFn || similarity;

  if (embeddingIds === null || embeddingIds.length === 0) {
    embeddingIds = Array.from({ length: embeddings.length }, (_, i) => i);
  }
  let fullEmbedMap = new Map(embeddingIds.map((value, i) => [value, i]));
  let embedMap = new Map(fullEmbedMap);
  let embedSimilarity: Map<any, number> = new Map();
  let score: number = Number.NEGATIVE_INFINITY;
  let highScoreId: any | null = null;

  for (let i = 0; i < embeddings.length; i++) {
    let emb = embeddings[i];
    let similarity = similarityFn(queryEmbedding, emb);
    embedSimilarity.set(embeddingIds[i], similarity);
    if (similarity * threshold > score) {
      highScoreId = embeddingIds[i];
      score = similarity * threshold;
    }
  }

  let results: [number, any][] = [];

  let embeddingLength = embeddings.length;
  let similarityTopKCount = similarityTopK || embeddingLength;

  while (results.length < Math.min(similarityTopKCount, embeddingLength)) {
    results.push([score, highScoreId]);
    embedMap.delete(highScoreId!);
    let recentEmbeddingId = highScoreId;
    score = Number.NEGATIVE_INFINITY;
    for (let embedId of Array.from(embedMap.keys())) {
      let overlapWithRecent = similarityFn(
        embeddings[embedMap.get(embedId)!],
        embeddings[fullEmbedMap.get(recentEmbeddingId!)!],
      );
      if (
        threshold * embedSimilarity.get(embedId)! -
        (1 - threshold) * overlapWithRecent >
        score
      ) {
        score =
          threshold * embedSimilarity.get(embedId)! -
          (1 - threshold) * overlapWithRecent;
        highScoreId = embedId;
      }
    }
  }

  let resultSimilarities = results.map(([s, _]) => s);
  let resultIds = results.map(([_, n]) => n);

  return [resultSimilarities, resultIds];
}

const LEARNER_MODES = new Set<VectorStoreQueryMode>([
  VectorStoreQueryMode.SVM,
  VectorStoreQueryMode.LINEAR_REGRESSION,
  VectorStoreQueryMode.LOGISTIC_REGRESSION,
]);

const MMR_MODE = VectorStoreQueryMode.MMR;

class SimpleVectorStoreData {
  embeddingDict: Record<string, number[]> = {};
  textIdToRefDocId: Record<string, string> = {};
}

export class SimpleVectorStore implements VectorStore {
  storesText: boolean = false;
  private data: SimpleVectorStoreData = new SimpleVectorStoreData();
  private fs: GenericFileSystem = defaultFS;
  private persistPath: string | undefined;

  constructor(data?: SimpleVectorStoreData, fs?: GenericFileSystem) {
    this.data = data || new SimpleVectorStoreData();
    this.fs = fs || defaultFS;
  }

  static async fromPersistDir(
    persistDir: string = DEFAULT_PERSIST_DIR,
    fs: GenericFileSystem = defaultFS,
  ): Promise<SimpleVectorStore> {
    let persistPath = `${persistDir}/vector_store.json`;
    return await SimpleVectorStore.fromPersistPath(persistPath, fs);
  }

  get client(): any {
    return null;
  }

  async get(textId: string): Promise<number[]> {
    return this.data.embeddingDict[textId];
  }

  async add(embeddingResults: BaseNode[]): Promise<string[]> {
    for (let node of embeddingResults) {
      this.data.embeddingDict[node.id_] = node.getEmbedding();

      if (!node.sourceNode) {
        continue;
      }

      this.data.textIdToRefDocId[node.id_] = node.sourceNode?.nodeId;
    }

    if (this.persistPath) {
      await this.persist(this.persistPath, this.fs);
    }

    return embeddingResults.map((result) => result.id_);
  }

  async delete(refDocId: string): Promise<void> {
    let textIdsToDelete = Object.keys(this.data.textIdToRefDocId).filter(
      (textId) => this.data.textIdToRefDocId[textId] === refDocId,
    );
    for (let textId of textIdsToDelete) {
      delete this.data.embeddingDict[textId];
      delete this.data.textIdToRefDocId[textId];
    }
    return Promise.resolve();
  }

  async query(query: VectorStoreQuery): Promise<VectorStoreQueryResult> {
    if (!_.isNil(query.filters)) {
      throw new Error(
        "Metadata filters not implemented for SimpleVectorStore yet.",
      );
    }

    let items = Object.entries(this.data.embeddingDict);

    let nodeIds: string[], embeddings: number[][];
    if (query.docIds) {
      let availableIds = new Set(query.docIds);
      const queriedItems = items.filter((item) => availableIds.has(item[0]));
      nodeIds = queriedItems.map((item) => item[0]);
      embeddings = queriedItems.map((item) => item[1]);
    } else {
      // No docIds specified, so use all available items
      nodeIds = items.map((item) => item[0]);
      embeddings = items.map((item) => item[1]);
    }

    let queryEmbedding = query.queryEmbedding!;

    let topSimilarities: number[], topIds: string[];
    if (LEARNER_MODES.has(query.mode)) {
      [topSimilarities, topIds] = getTopKEmbeddingsLearner(
        queryEmbedding,
        embeddings,
        query.similarityTopK,
        nodeIds,
      );
    } else if (query.mode === MMR_MODE) {
      let mmrThreshold = query.mmrThreshold;
      [topSimilarities, topIds] = getTopKMMREmbeddings(
        queryEmbedding,
        embeddings,
        null,
        query.similarityTopK,
        nodeIds,
        mmrThreshold,
      );
    } else if (query.mode === VectorStoreQueryMode.DEFAULT) {
      [topSimilarities, topIds] = getTopKEmbeddings(
        queryEmbedding,
        embeddings,
        query.similarityTopK,
        nodeIds,
      );
    } else {
      throw new Error(`Invalid query mode: ${query.mode}`);
    }

    return Promise.resolve({
      similarities: topSimilarities,
      ids: topIds,
    });
  }

  async persist(
    persistPath: string = `${DEFAULT_PERSIST_DIR}/vector_store.json`,
    fs?: GenericFileSystem,
  ): Promise<void> {
    fs = fs || this.fs;
    let dirPath = path.dirname(persistPath);
    if (!(await exists(fs, dirPath))) {
      await fs.mkdir(dirPath);
    }

    await fs.writeFile(persistPath, JSON.stringify(this.data));
  }

  static async fromPersistPath(
    persistPath: string,
    fs?: GenericFileSystem,
  ): Promise<SimpleVectorStore> {
    fs = fs || defaultFS;

    let dirPath = path.dirname(persistPath);
    if (!(await exists(fs, dirPath))) {
      await fs.mkdir(dirPath, { recursive: true });
    }

    let dataDict: any = {};
    try {
      let fileData = await fs.readFile(persistPath);
      dataDict = JSON.parse(fileData.toString());
    } catch (e) {
      console.error(
        `No valid data found at path: ${persistPath} starting new store.`,
      );
    }

    let data = new SimpleVectorStoreData();
    data.embeddingDict = dataDict.embeddingDict ?? {};
    data.textIdToRefDocId = dataDict.textIdToRefDocId ?? {};
    const store = new SimpleVectorStore(data);
    store.persistPath = persistPath;
    store.fs = fs;
    return store;
  }

  static fromDict(saveDict: SimpleVectorStoreData): SimpleVectorStore {
    let data = new SimpleVectorStoreData();
    data.embeddingDict = saveDict.embeddingDict;
    data.textIdToRefDocId = saveDict.textIdToRefDocId;
    return new SimpleVectorStore(data);
  }

  toDict(): SimpleVectorStoreData {
    return {
      embeddingDict: this.data.embeddingDict,
      textIdToRefDocId: this.data.textIdToRefDocId,
    };
  }
}
