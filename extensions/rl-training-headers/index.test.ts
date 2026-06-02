import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import register from "./index.js";

describe("rl-training-headers plugin", () => {
  const hooks: Record<string, Function> = {};
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
    for (const key of Object.keys(hooks)) {
      delete hooks[key];
    }
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("registers the before_prompt_build hook", () => {
    const api = createApi(hooks);

    register(api as any);

    expect(api.on).toHaveBeenCalledWith("before_prompt_build", expect.any(Function));
    expect(api.logger.info).toHaveBeenCalledWith(
      "rl-training-headers: activated (fetch patched)",
    );
  });

  it("keeps session headers isolated across concurrent runs", async () => {
    const seenSessionIds: string[] = [];
    const seenTurnTypes: string[] = [];
    globalThis.fetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const headers = new Headers(init?.headers);
      seenSessionIds.push(headers.get("X-Session-Id") ?? "");
      seenTurnTypes.push(headers.get("X-Turn-Type") ?? "");
      return new Response(null, { status: 200 });
    }) as typeof globalThis.fetch;

    register(createApi(hooks) as any);

    const runSession = async (sessionId: string, trigger: string) => {
      await Promise.resolve();
      hooks.before_prompt_build?.({}, { sessionId, trigger });
      await Promise.resolve();
      await globalThis.fetch("https://example.test/llm", { method: "POST" });
    };

    await Promise.all([
      runSession("session-a", "user"),
      runSession("session-b", "heartbeat"),
    ]);

    expect(seenSessionIds).toEqual(["session-a", "session-b"]);
    expect(seenTurnTypes).toEqual(["main", "side"]);
  });
});

function createApi(hooks: Record<string, Function>) {
  return {
    pluginConfig: {},
    logger: {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      debug: vi.fn(),
    },
    on: vi.fn((name: string, handler: Function) => {
      hooks[name] = handler;
    }),
  };
}
