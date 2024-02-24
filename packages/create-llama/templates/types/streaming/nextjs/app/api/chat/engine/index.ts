import { SimpleChatEngine } from "llamaindex/engines/chat/SimpleChatEngine";
import { LLM } from "llamaindex/llm/types";

export async function createChatEngine(llm: LLM) {
  return new SimpleChatEngine({
    llm,
  });
}
