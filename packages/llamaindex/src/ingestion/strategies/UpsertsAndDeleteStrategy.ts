import { BaseNode, TransformComponent } from "@llamaindex/core/schema";
import type { BaseVectorStore } from "@llamaindex/core/vector-store";
import type { BaseDocumentStore } from "../../storage/docStore/types.js";
import { classify } from "./classify.js";

/**
 * Handle docstore upserts by checking hashes and ids.
 * Identify missing docs and delete them from docstore and vector store
 */
export class UpsertsAndDeleteStrategy extends TransformComponent {
  protected docStore: BaseDocumentStore;
  protected vectorStores: BaseVectorStore[] | undefined;

  constructor(docStore: BaseDocumentStore, vectorStores?: BaseVectorStore[]) {
    super(async (nodes: BaseNode[]): Promise<BaseNode[]> => {
      const { dedupedNodes, missingDocs, unusedDocs } = await classify(
        this.docStore,
        nodes,
      );

      // remove unused docs
      for (const refDocId of unusedDocs) {
        await this.docStore.deleteRefDoc(refDocId, false);
        if (this.vectorStores) {
          for (const vectorStore of this.vectorStores) {
            await vectorStore.delete(refDocId);
          }
        }
      }

      // remove missing docs
      for (const docId of missingDocs) {
        await this.docStore.deleteDocument(docId, true);
        if (this.vectorStores) {
          for (const vectorStore of this.vectorStores) {
            await vectorStore.delete(docId);
          }
        }
      }

      await this.docStore.addDocuments(dedupedNodes, true);

      return dedupedNodes;
    });
    this.docStore = docStore;
    this.vectorStores = vectorStores;
  }
}
