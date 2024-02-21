import mammoth from "mammoth";
import { Document } from "llamaindex/Node";
import { defaultFS } from "llamaindex/env";
import type { GenericFileSystem } from "llamaindex";
import type { FileReader } from "./type";

export class DocxReader implements FileReader {
  /** DocxParser */
  async loadData(
    file: string,
    fs: GenericFileSystem = defaultFS,
  ): Promise<Document[]> {
    const dataBuffer = await fs.readRawFile(file);
    const { value } = await mammoth.extractRawText({ buffer: dataBuffer });
    return [new Document({ text: value, id_: file })];
  }
}
