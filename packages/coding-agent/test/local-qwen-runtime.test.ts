import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import {
	createLocalQwenModel,
	LocalQwenRuntime,
} from "../src/core/local-qwen-runtime.ts";
import { ModelRegistry } from "../src/core/model-registry.ts";

describe("LocalQwenRuntime", () => {
	let tempDir: string | undefined;

	afterEach(() => {
		if (tempDir) {
			rmSync(tempDir, { recursive: true, force: true });
			tempDir = undefined;
		}
	});

	test("requires no provider credentials", async () => {
		tempDir = mkdtempSync(join(tmpdir(), "pi-local-qwen-auth-"));
		const registry = ModelRegistry.create(
			AuthStorage.inMemory(),
			join(tempDir, "models.json"),
		);
		const model = createLocalQwenModel("/models/qwen3-8b");
		expect(registry.hasConfiguredAuth(model)).toBe(true);
		expect(await registry.getApiKeyAndHeaders(model)).toEqual({ ok: true });
	});

	test("uses one native worker transport for generation and RCG policy", async () => {
		tempDir = mkdtempSync(join(tmpdir(), "pi-local-qwen-worker-"));
		const workerPath = join(tempDir, "worker.mjs");
		writeFileSync(
			workerPath,
			`
import { createInterface } from "node:readline";
console.log(JSON.stringify({ id: "__ready__", ok: true, result: { rcgStep: 2000 } }));
const input = createInterface({ input: process.stdin });
input.on("line", (line) => {
  const request = JSON.parse(line);
  if (request.command === "generate") {
    console.log(JSON.stringify({
      id: request.id,
      ok: true,
      result: {
        text: "{\\"name\\":\\"read\\",\\"arguments\\":{\\"path\\":\\"README.md\\"}}",
        finishReason: "stop",
        promptTokens: 12,
        outputTokens: 8,
        cacheReadTokens: 10,
        cacheWriteTokens: 2
      }
    }));
  } else if (request.command === "plan") {
    console.log(JSON.stringify({
      id: request.id,
      ok: true,
      result: { actions: [{ id: "read-1", title: "read", objective: "Read the file" }] }
    }));
  }
});
`,
		);
		const runtime = new LocalQwenRuntime({
			pythonPath: process.execPath,
			workerPath,
			modelPath: "/models/qwen3-8b",
			rcgCheckpoint: "/checkpoints/rcg.pt",
			startupTimeoutMs: 5_000,
		});
		const model = createLocalQwenModel("/models/qwen3-8b");
		try {
			const message = await runtime
				.stream(
					model,
					{
						messages: [{ role: "user", content: "Read README", timestamp: 1 }],
					},
					{ reasoning: "off" },
				)
				.result();
			expect(message.stopReason).toBe("toolUse");
			expect(message.content).toContainEqual({
				type: "toolCall",
				id: expect.any(String),
				name: "read",
				arguments: { path: "README.md" },
			});
			expect(message.usage.cacheRead).toBe(10);

			const actions = await runtime.plan({
				task: "Read README",
				latestObservation: "",
				requestIndex: 0,
				maxBranches: 1,
				tools: [],
				residualMemory: "",
				globalState: {
					taskEpoch: 0,
					completedActions: [],
					toolResults: [],
					activeBranches: [],
				},
			});
			expect(actions[0]?.title).toBe("read");
		} finally {
			runtime.dispose();
		}
	});
});
