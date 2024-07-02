import type {
  ChatMessage,
  ChatResponse,
  ChatResponseChunk,
  LLMChatParamsNonStreaming,
  LLMChatParamsStreaming,
} from "@llamaindex/core/llms";
import { getEnv } from "@llamaindex/env";
import { Settings } from "../Settings.js";
import { type StreamCallbackResponse } from "../callbacks/CallbackManager.js";
import { BaseLLM } from "./base.js";

export const ALL_AVAILABLE_MISTRAL_MODELS = {
  "mistral-tiny": { contextWindow: 32000 },
  "mistral-small": { contextWindow: 32000 },
  "mistral-medium": { contextWindow: 32000 },
};

export class MistralAISession {
  apiKey?: string;
  private client: any;

  constructor(init?: Partial<MistralAISession>) {
    if (init?.apiKey) {
      this.apiKey = init?.apiKey;
    } else {
      this.apiKey = getEnv("MISTRAL_API_KEY");
    }
    if (!this.apiKey) {
      throw new Error("Set Mistral API key in MISTRAL_API_KEY env variable"); // Overriding MistralAI package's error message
    }
  }

  async getClient() {
    const { default: MistralClient } = await import("@mistralai/mistralai");
    if (!this.client) {
      this.client = new MistralClient(this.apiKey);
    }
    return this.client;
  }
}

/**
 * MistralAI LLM implementation
 */
export class MistralAI extends BaseLLM {
  // Per completion MistralAI params
  model: keyof typeof ALL_AVAILABLE_MISTRAL_MODELS;
  temperature: number;
  topP: number;
  maxTokens?: number;
  apiKey?: string;
  safeMode: boolean;
  randomSeed?: number;

  private session: MistralAISession;

  constructor(init?: Partial<MistralAI>) {
    super();
    this.model = init?.model ?? "mistral-small";
    this.temperature = init?.temperature ?? 0.1;
    this.topP = init?.topP ?? 1;
    this.maxTokens = init?.maxTokens ?? undefined;
    this.safeMode = init?.safeMode ?? false;
    this.randomSeed = init?.randomSeed ?? undefined;
    this.session = new MistralAISession(init);
  }

  get metadata() {
    return {
      model: this.model,
      temperature: this.temperature,
      topP: this.topP,
      maxTokens: this.maxTokens,
      contextWindow: ALL_AVAILABLE_MISTRAL_MODELS[this.model].contextWindow,
      tokenizer: undefined,
    };
  }

  private buildParams(messages: ChatMessage[]): any {
    return {
      model: this.model,
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      topP: this.topP,
      safeMode: this.safeMode,
      randomSeed: this.randomSeed,
      messages,
    };
  }

  chat(
    params: LLMChatParamsStreaming,
  ): Promise<AsyncIterable<ChatResponseChunk>>;
  chat(params: LLMChatParamsNonStreaming): Promise<ChatResponse>;
  async chat(
    params: LLMChatParamsNonStreaming | LLMChatParamsStreaming,
  ): Promise<ChatResponse | AsyncIterable<ChatResponseChunk>> {
    const { messages, stream } = params;
    // Streaming
    if (stream) {
      return this.streamChat(params);
    }
    // Non-streaming
    const client = await this.session.getClient();
    const response = await client.chat(this.buildParams(messages));
    const message = response.choices[0].message;
    return {
      raw: response,
      message,
    };
  }

  protected async *streamChat({
    messages,
  }: LLMChatParamsStreaming): AsyncIterable<ChatResponseChunk> {
    const client = await this.session.getClient();
    const chunkStream = await client.chatStream(this.buildParams(messages));

    //Indices
    let idx_counter: number = 0;
    for await (const part of chunkStream) {
      if (!part.choices.length) continue;

      part.choices[0].index = idx_counter;
      const isDone: boolean =
        part.choices[0].finish_reason === "stop" ? true : false;

      const stream_callback: StreamCallbackResponse = {
        index: idx_counter,
        isDone: isDone,
        token: part,
      };

      Settings.callbackManager.dispatchEvent("stream", stream_callback);

      idx_counter++;

      yield {
        raw: part,
        delta: part.choices[0].delta.content ?? "",
      };
    }
    return;
  }
}
