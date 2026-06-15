import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { randomUUID } from "node:crypto";
import { dirname } from "node:path";
import { createInterface } from "node:readline";
import type {
	AssistantMessage,
	AssistantMessageEventStream,
	Context,
	Model,
	SimpleStreamOptions,
	ThinkingContent,
	TextContent,
	ToolCall,
} from "@earendil-works/pi-ai";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import type {
	SpeculativeSwarmDynamicAction,
	SpeculativeSwarmDynamicPlanInput,
	SpeculativeSwarmDynamicPolicy,
	SpeculativeSwarmDynamicRoute,
	SpeculativeSwarmDynamicRouteInput,
	SpeculativeSwarmDynamicSelection,
	SpeculativeSwarmDynamicSelectionInput,
} from "./speculative-swarm.ts";

export const LOCAL_QWEN_API = "local-qwen-native";
export const LOCAL_QWEN_PROVIDER = "local-qwen";

export interface LocalQwenRuntimeOptions {
	pythonPath: string;
	workerPath: string;
	modelPath: string;
	rcgCheckpoint?: string;
	rcgDevice?: "cpu" | "mps";
	cacheSequences?: number;
	cacheBytes?: number;
	decodeConcurrency?: number;
	prefillConcurrency?: number;
	maxKvSize?: number;
	startupTimeoutMs?: number;
	mainMaxTokens?: number;
	branchMaxTokens?: number;
	orchestratorMaxTokens?: number;
}

interface WorkerResponse {
	id: string | null;
	ok: boolean;
	result?: unknown;
	error?: string;
}

interface GenerationResult {
	text: string;
	finishReason: "stop" | "length";
	promptTokens: number;
	outputTokens: number;
	cacheReadTokens: number;
	cacheWriteTokens: number;
	promptTps?: number;
	generationTps?: number;
	peakMemoryGb?: number;
	cacheEntries?: number;
	cacheBytes?: number;
}

interface PendingRequest {
	resolve: (value: unknown) => void;
	reject: (error: Error) => void;
}

export function createLocalQwenModel(modelPath: string): Model<any> {
	return {
		id: "qwen3-8b-local",
		name: "Qwen3-8B Local Native",
		api: LOCAL_QWEN_API,
		provider: LOCAL_QWEN_PROVIDER,
		baseUrl: `local://${modelPath}`,
		reasoning: true,
		input: ["text"],
		cost: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
		},
		contextWindow: 32_768,
		maxTokens: 512,
	};
}

export function isLocalQwenModel(model: Model<any> | undefined): boolean {
	return model?.api === LOCAL_QWEN_API && model.provider === LOCAL_QWEN_PROVIDER;
}

function zeroCost() {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		total: 0,
	};
}

function parseAssistantContent(raw: string): Array<TextContent | ThinkingContent | ToolCall> {
	const content: Array<TextContent | ThinkingContent | ToolCall> = [];
	let remaining = raw;
	const thinkPattern = /<think>([\s\S]*?)(?:<\/think>|$)/gi;
	for (const match of raw.matchAll(thinkPattern)) {
		const thinking = match[1]?.trim();
		if (thinking) {
			content.push({ type: "thinking", thinking });
		}
	}
	remaining = remaining.replace(thinkPattern, "");

	const toolPattern = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/gi;
	const toolCalls: ToolCall[] = [];
	for (const match of remaining.matchAll(toolPattern)) {
		try {
			const parsed = JSON.parse(match[1]) as {
				name?: string;
				arguments?: Record<string, unknown> | string;
			};
			const args =
				typeof parsed.arguments === "string"
					? (JSON.parse(parsed.arguments) as Record<string, unknown>)
					: (parsed.arguments ?? {});
			if (parsed.name) {
				toolCalls.push({
					type: "toolCall",
					id: `local-qwen-tool-${randomUUID()}`,
					name: parsed.name,
					arguments: args,
				});
			}
		} catch {
			// Keep malformed calls visible so the agent can report and recover.
		}
	}
	remaining = remaining.replace(toolPattern, "").trim();
	if (toolCalls.length === 0 && remaining.startsWith("{") && remaining.endsWith("}")) {
		try {
			const parsed = JSON.parse(remaining) as {
				name?: string;
				arguments?: Record<string, unknown> | string;
			};
			if (parsed.name && parsed.arguments !== undefined) {
				toolCalls.push({
					type: "toolCall",
					id: `local-qwen-tool-${randomUUID()}`,
					name: parsed.name,
					arguments:
						typeof parsed.arguments === "string"
							? (JSON.parse(parsed.arguments) as Record<string, unknown>)
							: parsed.arguments,
				});
				remaining = "";
			}
		} catch {
			// A normal JSON answer remains plain text unless it has tool-call shape.
		}
	}
	if (remaining) {
		content.push({ type: "text", text: remaining });
	}
	content.push(...toolCalls);
	if (content.length === 0) {
		content.push({ type: "text", text: raw.trim() });
	}
	return content;
}

export class LocalQwenRuntime implements SpeculativeSwarmDynamicPolicy {
	private readonly options: LocalQwenRuntimeOptions;
	private child?: ChildProcessWithoutNullStreams;
	private readonly pending = new Map<string, PendingRequest>();
	private nextId = 1;
	private ready?: Promise<void>;
	private readyResolve?: () => void;
	private readyReject?: (error: Error) => void;
	private exitHandler?: () => void;

	constructor(options: LocalQwenRuntimeOptions) {
		this.options = options;
	}

	stream(model: Model<any>, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream {
		const stream = createAssistantMessageEventStream();
		const startedAt = Date.now();
		const partial: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: zeroCost(),
			},
			stopReason: "stop",
			timestamp: startedAt,
		};
		stream.push({ type: "start", partial });

		void (async () => {
			try {
				const sessionId = options?.sessionId ?? "";
				const recentToolResult = context.messages
					.slice(-4)
					.some((message) => message.role === "toolResult");
				const tokenCap = sessionId.startsWith("swarm-branch-")
					? recentToolResult
						? Math.min(this.options.branchMaxTokens ?? 128, 64)
						: (this.options.branchMaxTokens ?? 128)
					: sessionId.startsWith("swarm-")
						? (this.options.orchestratorMaxTokens ?? 96)
						: (this.options.mainMaxTokens ?? 512);
				const result = await this.request<GenerationResult>(
					"generate",
					{
						context,
						options: {
							...options,
							signal: undefined,
							maxTokens: Math.min(options?.maxTokens ?? tokenCap, tokenCap),
						},
					},
					options?.signal,
				);
				const content = parseAssistantContent(result.text);
				const message: AssistantMessage = {
					...partial,
					content,
					usage: {
						input: result.promptTokens,
						output: result.outputTokens,
						cacheRead: result.cacheReadTokens,
						cacheWrite: result.cacheWriteTokens,
						totalTokens: result.promptTokens + result.outputTokens,
						cost: zeroCost(),
					},
					stopReason: content.some((item) => item.type === "toolCall")
						? "toolUse"
						: result.finishReason,
				};
				for (let index = 0; index < content.length; index++) {
					const item = content[index];
					if (item.type === "thinking") {
						stream.push({ type: "thinking_start", contentIndex: index, partial: message });
						stream.push({
							type: "thinking_delta",
							contentIndex: index,
							delta: item.thinking,
							partial: message,
						});
						stream.push({
							type: "thinking_end",
							contentIndex: index,
							content: item.thinking,
							partial: message,
						});
					} else if (item.type === "text") {
						stream.push({ type: "text_start", contentIndex: index, partial: message });
						stream.push({
							type: "text_delta",
							contentIndex: index,
							delta: item.text,
							partial: message,
						});
						stream.push({
							type: "text_end",
							contentIndex: index,
							content: item.text,
							partial: message,
						});
					} else {
						stream.push({ type: "toolcall_start", contentIndex: index, partial: message });
						stream.push({
							type: "toolcall_delta",
							contentIndex: index,
							delta: JSON.stringify({ name: item.name, arguments: item.arguments }),
							partial: message,
						});
						stream.push({
							type: "toolcall_end",
							contentIndex: index,
							toolCall: item,
							partial: message,
						});
					}
				}
				stream.push({
					type: "done",
					reason: message.stopReason as "stop" | "length" | "toolUse",
					message,
				});
			} catch (error) {
				const aborted = options?.signal?.aborted === true;
				const message: AssistantMessage = {
					...partial,
					stopReason: aborted ? "aborted" : "error",
					errorMessage: error instanceof Error ? error.message : String(error),
				};
				stream.push({
					type: "error",
					reason: message.stopReason as "aborted" | "error",
					error: message,
				});
			}
		})();
		return stream;
	}

	async plan(input: SpeculativeSwarmDynamicPlanInput): Promise<SpeculativeSwarmDynamicAction[]> {
		const response = await this.request<{ actions: SpeculativeSwarmDynamicAction[] }>("plan", {
			payload: input,
		});
		return response.actions;
	}

	select(input: SpeculativeSwarmDynamicSelectionInput): Promise<SpeculativeSwarmDynamicSelection> {
		return this.request("select", { payload: input });
	}

	route(input: SpeculativeSwarmDynamicRouteInput): Promise<SpeculativeSwarmDynamicRoute> {
		return this.request("route", { payload: input });
	}

	stats(): Promise<Record<string, unknown>> {
		return this.request("stats", {});
	}

	dispose(): void {
		if (this.exitHandler) {
			process.off("exit", this.exitHandler);
			this.exitHandler = undefined;
		}
		this.child?.kill("SIGTERM");
		this.child = undefined;
	}

	private async request<T>(command: string, payload: Record<string, unknown>, signal?: AbortSignal): Promise<T> {
		await this.ensureStarted();
		if (signal?.aborted) {
			throw new Error("Local Qwen request aborted");
		}
		const id = `local-qwen-${this.nextId++}`;
		return new Promise<T>((resolve, reject) => {
			const abort = () => {
				this.pending.delete(id);
				reject(new Error("Local Qwen request aborted"));
			};
			signal?.addEventListener("abort", abort, { once: true });
			this.pending.set(id, {
				resolve: (value) => {
					signal?.removeEventListener("abort", abort);
					resolve(value as T);
				},
				reject: (error) => {
					signal?.removeEventListener("abort", abort);
					reject(error);
				},
			});
			this.child!.stdin.write(`${JSON.stringify({ id, command, ...payload })}\n`);
		});
	}

	private ensureStarted(): Promise<void> {
		if (this.ready) {
			return this.ready;
		}
		this.ready = new Promise<void>((resolve, reject) => {
			this.readyResolve = resolve;
			this.readyReject = reject;
		});
		const args = [
			this.options.workerPath,
			"--model-path",
			this.options.modelPath,
			"--rcg-device",
			this.options.rcgDevice ?? "cpu",
			"--cache-sequences",
			String(this.options.cacheSequences ?? 64),
			"--cache-bytes",
			String(this.options.cacheBytes ?? 8 * 1024 ** 3),
			"--decode-concurrency",
			String(this.options.decodeConcurrency ?? 8),
			"--prefill-concurrency",
			String(this.options.prefillConcurrency ?? 4),
			"--max-output-tokens",
			String(this.options.mainMaxTokens ?? 512),
		];
		if (this.options.rcgCheckpoint) {
			args.push("--rcg-checkpoint", this.options.rcgCheckpoint);
		}
		if (this.options.maxKvSize) {
			args.push("--max-kv-size", String(this.options.maxKvSize));
		}
		this.child = spawn(this.options.pythonPath, args, {
			cwd: dirname(this.options.workerPath),
			stdio: ["pipe", "pipe", "pipe"],
		});
		const timeout = setTimeout(() => {
			this.readyReject?.(
				new Error(
					`Local Qwen worker did not start within ${this.options.startupTimeoutMs ?? 120_000}ms`,
				),
			);
			this.dispose();
		}, this.options.startupTimeoutMs ?? 120_000);
		const stdout = createInterface({ input: this.child.stdout });
		stdout.on("line", (line) => this.handleLine(line, timeout));
		this.child.stderr.on("data", (chunk) => {
			process.stderr.write(chunk);
		});
		this.child.on("error", (error) => this.failAll(error));
		this.child.on("exit", (code, signal) => {
			this.failAll(new Error(`Local Qwen worker exited code=${code ?? "null"} signal=${signal ?? "null"}`));
		});
		this.exitHandler = () => this.child?.kill("SIGTERM");
		process.once("exit", this.exitHandler);
		return this.ready;
	}

	private handleLine(line: string, startupTimeout: NodeJS.Timeout): void {
		let response: WorkerResponse;
		try {
			response = JSON.parse(line) as WorkerResponse;
		} catch {
			this.failAll(new Error(`Invalid local Qwen worker response: ${line.slice(0, 500)}`));
			return;
		}
		if (response.id === "__ready__") {
			clearTimeout(startupTimeout);
			if (response.ok) {
				this.readyResolve?.();
			} else {
				this.readyReject?.(new Error(response.error ?? "Local Qwen worker failed to initialize"));
			}
			return;
		}
		if (!response.id) {
			return;
		}
		const pending = this.pending.get(response.id);
		if (!pending) {
			return;
		}
		this.pending.delete(response.id);
		if (response.ok) {
			pending.resolve(response.result);
		} else {
			pending.reject(new Error(response.error ?? "Local Qwen worker request failed"));
		}
	}

	private failAll(error: Error): void {
		this.readyReject?.(error);
		for (const pending of this.pending.values()) {
			pending.reject(error);
		}
		this.pending.clear();
	}
}
