import type { BaseNode } from "@llamaindex/core/schema";
import {
  VectorStoreBase,
  type VectorStoreNoEmbedModel,
  type VectorStoreQuery,
  type VectorStoreQueryResult
} from './types.js';

import { QdrantClient, type QdrantClientParams, type Schemas } from "@qdrant/qdrant-js";
import { metadataDictToNode, nodeToMetadata } from "./utils.js";
import type { BaseEmbedding } from '@llamaindex/core/embeddings';

type PointStruct = {
  id: string;
  payload: Record<string, string>;
  vector: number[];
};

type QdrantVectorStoreOptions = (QdrantClientParams | { client?: QdrantClient; }) & {
  batchSize?: number;
  collectionName?: string;
  embedModel?: BaseEmbedding;
};

type QuerySearchResult = {
  id: string;
  score: number;
  payload: Record<string, unknown>;
  vector: number[] | null;
  version: number;
};

export class QdrantVectorStore
  extends VectorStoreBase
  implements VectorStoreNoEmbedModel
{
  storesText: boolean = true;

  batchSize: number;
  collectionName: string;

  private db: QdrantClient;
  private collectionInitialized: boolean = false;

  /**
   * Creates a new QdrantVectorStore.
   * @param collectionName Qdrant collection name
   * @param client Qdrant client
   * @param url Qdrant URL
   * @param apiKey Qdrant API key
   * @param batchSize Number of vectors to upload in a single batch
   * @param embedModel Embedding model
   */
  constructor(options: QdrantVectorStoreOptions) {
    const { embedModel, batchSize, collectionName } = options;
    super(embedModel);
    if ('client' in options && options.client) {
      this.db = options.client;
    } else {
      this.db = new QdrantClient(options as QdrantClientParams);
    }

    this.collectionName = collectionName ?? "default";
    this.batchSize = batchSize ?? 100;
  }

  client() {
    return this.db;
  }

  /**
   * Creates a collection in Qdrant.
   * @param collectionName Qdrant collection name
   * @param vectorSize Dimensionality of the vectors
   */
  async createCollection(collectionName: string, vectorSize: number) {
    await this.db.createCollection(collectionName, {
      vectors: {
        size: vectorSize,
        distance: "Cosine",
      },
    });
  }

  /**
   * Checks if the collection exists in Qdrant and creates it if not.
   * @param collectionName Qdrant collection name
   * @returns
   */
  async collectionExists(collectionName: string): Promise<boolean> {
    try {
      await this.db.getCollection(collectionName);
      return true;
    } catch (e) {
      return false;
    }
  }

  /**
   * Initializes the collection in Qdrant.
   * @param vectorSize Dimensionality of the vectors
   */
  async initializeCollection(vectorSize: number) {
    const exists = await this.collectionExists(this.collectionName);
    if (!exists) {
      await this.createCollection(this.collectionName, vectorSize);
    }
    this.collectionInitialized = true;
  }

  /**
   * Builds a list of points from the given nodes.
   * @param nodes
   * @returns
   */
  async buildPoints(nodes: BaseNode[]): Promise<{
    points: PointStruct[];
    ids: string[];
  }> {
    const points: PointStruct[] = [];
    const ids = [];

    for (let i = 0; i < nodes.length; i++) {
      const nodeIds = [];
      const vectors = [];
      const payloads = [];

      for (let j = 0; j < this.batchSize && i < nodes.length; j++, i++) {
        const node = nodes[i];

        nodeIds.push(node);

        vectors.push(node.getEmbedding());

        const metadata = nodeToMetadata(node);

        payloads.push(metadata);
      }

      for (let k = 0; k < nodeIds.length; k++) {
        const point: PointStruct = {
          id: nodeIds[k].id_,
          payload: payloads[k],
          vector: vectors[k],
        };

        points.push(point);
      }

      ids.push(...nodeIds.map((node) => node.id_));
    }

    return {
      points: points,
      ids: ids,
    };
  }

  /**
   * Adds the given nodes to the vector store.
   * @param embeddingResults List of nodes
   * @returns List of node IDs
   */
  async add(embeddingResults: BaseNode[]): Promise<string[]> {
    if (embeddingResults.length > 0 && !this.collectionInitialized) {
      await this.initializeCollection(
        embeddingResults[0].getEmbedding().length,
      );
    }

    const { points, ids } = await this.buildPoints(embeddingResults);

    const batchUpsert = async (points: PointStruct[]) => {
      await this.db.upsert(this.collectionName, {
        points: points,
      });
    };

    for (let i = 0; i < points.length; i += this.batchSize) {
      const chunk = points.slice(i, i + this.batchSize);
      await batchUpsert(chunk);
    }

    return ids;
  }

  /**
   * Deletes the given nodes from the vector store.
   * @param refDocId Node ID
   */
  async delete(refDocId: string): Promise<void> {
    const mustFilter = [
      {
        key: "doc_id",
        match: {
          value: refDocId,
        },
      },
    ];

    await this.db.delete(this.collectionName, {
      filter: {
        must: mustFilter,
      },
    });
  }

  /**
   * Converts the result of a query to a VectorStoreQueryResult.
   * @param response Query response
   * @returns VectorStoreQueryResult
   */
  private parseToQueryResult(
    response: Array<QuerySearchResult>,
  ): VectorStoreQueryResult {
    const nodes = [];
    const similarities = [];
    const ids = [];

    for (let i = 0; i < response.length; i++) {
      const item = response[i];
      const payload = item.payload;

      const node = metadataDictToNode(payload);

      ids.push(item.id);
      nodes.push(node);
      similarities.push(item.score);
    }

    return {
      nodes: nodes,
      similarities: similarities,
      ids: ids,
    };
  }

  /**
   * Queries the vector store for the closest matching data to the query embeddings.
   * @param query The VectorStoreQuery to be used
   * @param options Required by VectorStore interface.  Currently ignored.
   * @returns Zero or more Document instances with data from the vector store.
   */
  async query(
    query: VectorStoreQuery,
    options?: {
      qdrantFilters?: Schemas['Filter'];
    },
  ): Promise<VectorStoreQueryResult> {
    const qdrantFilters = options?.qdrantFilters;

    let queryFilters;

    if (!query.queryEmbedding) {
      throw new Error("No query embedding provided");
    }

    if (qdrantFilters) {
      queryFilters = qdrantFilters;
    } else {
      queryFilters = await this.buildQueryFilter(query);
    }

    const result = (await this.db.search(this.collectionName, {
      vector: query.queryEmbedding,
      limit: query.similarityTopK,
      ...(queryFilters && { filter: queryFilters }),
    })) as Array<QuerySearchResult>;

    return this.parseToQueryResult(result);
  }

  private buildQueryFilter(query: VectorStoreQuery): Schemas['Filter'] | null {
    if (!query.docIds && !query.queryStr && !query.filters) {
      return null;
    }

    const mustConditions = [];

    if (query.docIds) {
      mustConditions.push({
        key: "doc_id",
        match: {
          any: query.docIds,
        },
      });
    }

    if (!query.filters) {
      return {
        must: mustConditions,
      };
    }

    const metadataFilters = query.filters.filters;

    for (let i = 0; i < metadataFilters.length; i++) {
      const filter = metadataFilters[i];

      mustConditions.push({
        key: filter.key,
        match: {
          value: filter.value,
        },
      });
    }

    return {
      must: mustConditions,
    };
  }
}
