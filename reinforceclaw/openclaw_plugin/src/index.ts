// reinforceclaw openclaw plugin: keeps the Python bridge alive inside the user's gateway.

import { spawn, type ChildProcess } from "node:child_process";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

const pending = new Map<string, { prompt: string; channel: string; context: unknown }>();
let bridge: ChildProcess | undefined;
let bridgeSecret = "";
const SAFE_COMMANDS = new Set(["good", "bad", "undo", "status"]);
const ADMIN_COMMANDS = new Set(["train", "rollback", "reset", "on", "off"]);
const MAX_PENDING = 1000;
const DEFAULT_HOST = "http://127.0.0.1:8420";

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function sessionKey(event: any, ctx?: any) {
  return String(
    ctx?.sessionKey ??
    event?.sessionKey ??
    event?.conversationId ??
    event?.threadId ??
    event?.target ??
    channelOf(event) ??
    "default"
  );
}

function contentOf(event: any) {
  return String(event?.content ?? event?.message ?? event?.text ?? "");
}

function channelOf(event: any) {
  return String(
    event?.channelId ?? event?.channel?.id ?? event?.channel?.name ??
    event?.channel ?? event?.metadata?.provider ?? "openclaw"
  );
}

function rolloutContext(event: any, ctx: any, prompt: string) {
  const raw = Array.isArray(ctx?.messages) ? ctx.messages : Array.isArray(event?.messages) ? event.messages : [];
  const messages = raw
    .map((m: any) => ({ role: String(m?.role || ""), content: contentOf(m).trim() }))
    .filter((m: any) => ["system", "user", "assistant"].includes(m.role) && m.content)
    .slice(-12);
  if (!messages.some((m: any) => m.role === "user" && m.content === prompt)) {
    messages.push({ role: "user", content: prompt });
  }
  return { messages, session: sessionKey(event, ctx), channel: channelOf(event) };
}

function commandParts(call: any[]) {
  const first = call[0] ?? {};
  const args = Array.isArray(first) ? first : Array.isArray(first.args) ? first.args : [];
  return { args, ctx: call[1] ?? first };
}

function secretHeader() {
  const secret = bridgeSecret || process.env.REINFORCECLAW_OPENCLAW_SECRET;
  return secret ? { "X-ReinforceClaw-Secret": secret } : {};
}

function bridgeEnv(secret: string) {
  const env: Record<string, string> = { ...(process.env as Record<string, string>) };
  for (const key of Object.keys(env)) {
    if (key.endsWith("_API_KEY") || ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"].includes(key)) {
      delete env[key];
    }
  }
  env.REINFORCECLAW_OPENCLAW_SECRET = secret;
  return env;
}

function pluginConfig(api: any) {
  return ((api as any).pluginConfig ?? {}) as Record<string, unknown>;
}

function secretFrom(cfg: Record<string, unknown>) {
  return String(cfg.reinforceclawSecret || bridgeFile().secret || process.env.REINFORCECLAW_OPENCLAW_SECRET || "");
}

function bridgeFile(): Record<string, string> {
  try {
    return JSON.parse(readFileSync(join(homedir(), ".reinforceclaw", "openclaw_bridge.json"), "utf8"));
  } catch {
    return {};
  }
}

function localHost(value: unknown) {
  const raw = String(value || bridgeFile().host || DEFAULT_HOST);
  try {
    const url = new URL(raw);
    if (url.protocol === "http:" && ["127.0.0.1", "localhost", "::1", "[::1]"].includes(url.hostname)) {
      return url.origin;
    }
  } catch {}
  return DEFAULT_HOST;
}

function callerId(ctx: any) {
  return String(ctx?.userId ?? ctx?.user?.id ?? ctx?.sender?.id ?? ctx?.author?.id ?? "");
}

function adminAllowed(sub: string, ctx: any, cfg: Record<string, unknown>) {
  if (!ADMIN_COMMANDS.has(sub) || ![true, "true", "1"].includes(cfg.reinforceclawAllowAdminCommands as any)) return false;
  const allowed = String(cfg.reinforceclawAdminUsers || "")
    .split(",").map((x) => x.trim()).filter(Boolean);
  return allowed.length > 0 && allowed.includes(callerId(ctx));
}

async function bridgeReady(host: string) {
  try {
    const r = await fetch(`${host}/feedback/status`, { headers: secretHeader() });
    return r.ok;
  } catch {
    return false;
  }
}

async function waitForBridge(host: string) {
  for (let i = 0; i < 10; i++) {
    if (await bridgeReady(host)) return true;
    await sleep(200);
  }
  return false;
}

function post(host: string, path: string, body: any) {
  return fetch(`${host}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...secretHeader(),
    },
    body: JSON.stringify(body),
  });
}

async function command(host: string, body: any) {
  try {
    const r = await post(host, "/feedback/command", body);
    const text = await r.text();
    let payload: any = {};
    try { payload = text ? JSON.parse(text) : {}; } catch {}
    const message = payload.message;
    if (!r.ok) return message || text || `error: ${r.status}`;
    return typeof message === "string" ? message : JSON.stringify(message);
  } catch {
    return "reinforceclaw server not running";
  }
}

export default definePluginEntry({
  id: "reinforceclaw-feedback",
  name: "ReinforceClaw Feedback",

  register(api) {
    const cfg = pluginConfig(api);
    const host = localHost(cfg.reinforceclawHost);
    const secret = secretFrom(cfg);
    bridgeSecret = secret;

    const startBridge = async () => {
      if (await bridgeReady(host)) return;
      if (!bridge) {
        const bridgeCfg = bridgeFile();
        const python = String(cfg.reinforceclawPython || bridgeCfg.python || process.env.REINFORCECLAW_PYTHON || "python3");
        bridge = spawn(python, ["-m", "reinforceclaw.hooks.openclaw"], {
          env: bridgeEnv(secret),
          stdio: "ignore",
        });
        bridge.on("exit", () => { bridge = undefined; });
        bridge.on("error", () => { bridge = undefined; });
      }
      await waitForBridge(host);
    };

    if (api.registerService) api.registerService({
      id: "reinforceclaw-bridge",
      start: startBridge,
      stop: () => {
        bridge?.kill();
        bridge = undefined;
      },
    });
    else void startBridge();

    const on = (names: string[], handler: (...args: any[]) => Promise<void>) => {
      for (const name of names) {
        try { api.on(name, handler); } catch {}
      }
    };

    // Stash every user message so the outbound hook can pair it with the bot response.
    on(["message_received", "message:received"], async (event: any, ctx: any) => {
      const prompt = contentOf(event).trim();
      if (!prompt) return;
      const key = sessionKey(event, ctx);
      if (pending.size >= MAX_PENDING) {
        const oldest = pending.keys().next().value;
        if (oldest !== undefined) pending.delete(oldest);
      }
      pending.set(key, {
        prompt,
        channel: channelOf(event),
        context: rolloutContext(event, ctx, prompt),
      });
    });

    // Pair each bot reply with the user's message and send it to Python.
    on([
      "message_sending", "message:sending", "message_sent", "message:sent",
    ], async (event: any, ctx: any) => {
      const key = sessionKey(event, ctx);
      const user = pending.get(key);
      const response = contentOf(event).trim();
      if (!user || !response) return;

      try {
        await startBridge();
        const r = await post(host, "/feedback/capture", {
          sessionKey: key,
          prompt: user.prompt,
          response,
          channel: user.channel,
          context: user.context,
        });
        if (r.ok) pending.delete(key);
      } catch (e: any) {
        console.error("[reinforceclaw]", e.message);
      }
    });

    const registerRateCommand = (name: string) => api.registerCommand({
      name,
      acceptsArgs: true,
      requireAuth: true,
      description: "Rate ReinforceClaw: good, bad, undo, status",
      handler: async (...call: any[]) => {
        const { args, ctx } = commandParts(call);
        const sub = String(args[0] || "").toLowerCase();
        if (SAFE_COMMANDS.has(sub) || adminAllowed(sub, ctx, cfg)) {
          await startBridge();
          return { text: await command(host, {
            sessionKey: sessionKey(ctx, ctx),
            command: sub,
          }) };
        }
        return { text: `/${name} good | bad | undo | status` };
      },
    });
    registerRateCommand("rl");
    registerRateCommand("rc");
    registerRateCommand("reinforceclaw");
  },
});
