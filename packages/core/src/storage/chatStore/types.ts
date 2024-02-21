// todo(alex): merge type
export type MessageType =
  | "user"
  | "assistant"
  | "system"
  | "generic"
  | "function"
  | "memory"
  | "tool";

export interface ChatMessage {
  // TODO: use MessageContent
  content: any;
  role: MessageType;
  additionalKwargs?: Record<string, any>;
}
export interface BaseChatStore {
  setMessages(key: string, messages: ChatMessage[]): void;
  getMessages(key: string): ChatMessage[];
  addMessage(key: string, message: ChatMessage): void;
  deleteMessages(key: string): ChatMessage[] | null;
  deleteMessage(key: string, idx: number): ChatMessage | null;
  deleteLastMessage(key: string): ChatMessage | null;
  getKeys(): string[];
}
