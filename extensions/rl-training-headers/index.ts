import { AsyncLocalStorage } from "node:async_hooks";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

type RlTrainingConfig = {
  sessionIdHeader: string;
  turnTypeHeader: string;
};

function resolveConfig(api: OpenClawPluginApi): RlTrainingConfig {
  const cfg = (api.pluginConfig ?? {}) as Partial<RlTrainingConfig>;
  return {
    sessionIdHeader: cfg.sessionIdHeader ?? "X-Session-Id",
    turnTypeHeader: cfg.turnTypeHeader ?? "X-Turn-Type",
  };
}

// Triggers classified as "side" (non-user-facing housekeeping runs).
const SIDE_TRIGGERS = new Set(["heartbeat", "memory", "cron"]);

export default function register(api: OpenClawPluginApi) {
  const config = resolveConfig(api);
  const headerStore = new AsyncLocalStorage<Record<string, string>>();

  const originalFetch = globalThis.fetch;

  globalThis.fetch = function rlPatchedFetch(
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> {
    const scopedHeaders = headerStore.getStore();
    if (scopedHeaders && init?.method?.toUpperCase() === "POST") {
      const merged = new Headers(init.headers);
      for (const [k, v] of Object.entries(scopedHeaders)) {
        // Plugin headers go first; per-request headers can still override.
        if (!merged.has(k)) {
          merged.set(k, v);
        }
      }
      return originalFetch.call(globalThis, input, { ...init, headers: merged });
    }
    return originalFetch.call(globalThis, input, init);
  } as typeof globalThis.fetch;

  api.on("before_prompt_build", (_event, ctx) => {
    const sessionId = ctx.sessionId ?? "";
    const turnType = SIDE_TRIGGERS.has(ctx.trigger ?? "") ? "side" : "main";
    headerStore.enterWith({
      [config.sessionIdHeader]: sessionId,
      [config.turnTypeHeader]: turnType,
    });
    return {};
  });

  api.logger.info("rl-training-headers: activated (fetch patched)");
}
