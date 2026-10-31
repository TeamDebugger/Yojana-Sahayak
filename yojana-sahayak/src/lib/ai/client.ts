
import { GoogleGenAI } from "@google/genai";

// Server-side only. Never expose this key to the browser.
export function llmConfigured(): boolean {
  return Boolean(process.env.GEMINI_API_KEY && process.env.LLM_MODEL);
}

export const MAX_OUTPUT_TOKENS = 2048;

export type StreamPart = { delta: string } | { stop: string | null };

export async function* streamAnswer(
  system: string,
  messages: { role: "user" | "assistant"; content: string }[],
  signal?: AbortSignal,
): AsyncGenerator<StreamPart> {
  if (signal?.aborted) return;

  const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY!,
  });

  const stream = await ai.models.generateContentStream({
    model: process.env.LLM_MODEL!,
    contents: messages.map((message) => ({
      role: message.role === "assistant" ? "model" : "user",
      parts: [{ text: message.content }],
    })),
    config: {
      systemInstruction: system,
      maxOutputTokens: MAX_OUTPUT_TOKENS,
    },
  });

  let stop: string | null = null;

  for await (const chunk of stream) {
    if (signal?.aborted) return;

    const text = chunk.text;
    if (text) {
      yield { delta: text };
    }

    if (chunk.candidates?.some(
      (candidate) => candidate.finishReason === "MAX_TOKENS",
    )) {
      stop = "max_tokens";
    }
  }

  if (!signal?.aborted) {
    yield { stop };
  }
}
