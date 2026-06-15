import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage, AgentTool } from "@earendil-works/pi-agent-core";
import {
	type AssistantMessage,
	type AssistantMessageEvent,
	EventStream,
	type Message,
	type Model,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { afterEach, describe, expect, it } from "vitest";
import {
	createSpeculativeSwarmController,
	extractActionCandidates,
	extractSequentialSteps,
	isLikelySequentialTask,
} from "../src/core/speculative-swarm.ts";

class MockAssistantStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event type");
			},
		);
	}
}

function createModel(): Model<"openai-responses"> {
	return {
		id: "mock",
		name: "mock",
		api: "openai-responses",
		provider: "openai",
		baseUrl: "https://example.invalid",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 8192,
		maxTokens: 2048,
	};
}

function createAssistantMessage(
	content: AssistantMessage["content"],
	stopReason: AssistantMessage["stopReason"] = "stop",
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "openai-responses",
		provider: "openai",
		model: "mock",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason,
		timestamp: Date.now(),
	};
}

function identityConverter(messages: AgentMessage[]): Message[] {
	return messages.filter(
		(message) => message.role === "user" || message.role === "assistant" || message.role === "toolResult",
	) as Message[];
}

function createTool(name: string, result: string): AgentTool {
	const parameters = Type.Object({ value: Type.Optional(Type.String()) });
	return {
		name,
		label: name,
		description: `Execute ${name} for one sequence step`,
		parameters,
		async execute() {
			return {
				content: [{ type: "text", text: result }],
				details: { result },
			};
		},
	};
}

describe("speculative swarm sequential cooperation", () => {
	const tempDirs: string[] = [];
	const previousTraceFile = process.env.PI_SWARM_TRACE_FILE;

	afterEach(() => {
		if (previousTraceFile === undefined) {
			delete process.env.PI_SWARM_TRACE_FILE;
		} else {
			process.env.PI_SWARM_TRACE_FILE = previousTraceFile;
		}
		for (const dir of tempDirs.splice(0)) {
			rmSync(dir, { recursive: true, force: true });
		}
	});

	it("detects and pre-splits ordered query actions", () => {
		const query =
			"First look up the parent kanji, then find the radical meaning, and finally list kanji using that meaning.";
		expect(isLikelySequentialTask(query)).toBe(true);
		expect(extractSequentialSteps(query)).toEqual([
			"look up the parent kanji",
			"find the radical meaning",
			"list kanji using that meaning.",
		]);
		expect(extractActionCandidates(query)).toHaveLength(3);
	});

	it("extracts unordered independent actions without selecting a parallel mode", () => {
		const query = "Fetch weather for London, search flights to Paris, and check the EUR exchange rate.";
		expect(extractActionCandidates(query)).toEqual([
			"Fetch weather for London",
			"search flights to Paris",
			"check the EUR exchange rate",
		]);
	});

	it("continues the same B and C sessions after upstream tool results arrive", async () => {
		const tempDir = mkdtempSync(join(tmpdir(), "pi-swarm-test-"));
		tempDirs.push(tempDir);
		const traceFile = join(tempDir, "trace.jsonl");
		process.env.PI_SWARM_TRACE_FILE = traceFile;

		const model = createModel();
		let dynamicPlanCalls = 0;
		let dynamicSelectCalls = 0;
		const callsBySession = new Map<string, number>();
		const seenContexts = new Map<string, string[]>();
		const streamFn = (_model: Model<any>, context: { messages: Message[] }, options?: { sessionId?: string }) => {
			const sessionId = options?.sessionId ?? "unknown";
			const call = (callsBySession.get(sessionId) ?? 0) + 1;
			callsBySession.set(sessionId, call);
			const serialized = JSON.stringify(context.messages);
			const contexts = seenContexts.get(sessionId) ?? [];
			contexts.push(serialized);
			seenContexts.set(sessionId, contexts);

			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				let message: AssistantMessage;
				if (sessionId.includes("Turn0OrchestratorBrief")) {
					message = createAssistantMessage([
						{
							type: "text",
							text: "BRANCH_PROTOTYPES:\n- branch_id: all\n  vote_for_when: evidence\n  reject_when: unsupported\n  must_verify: tools\nCOLLECTIVE_DECISION_RUBRIC:\n- prefer evidence",
						},
					]);
				} else if (sessionId.endsWith("-A") && call === 1) {
					message = createAssistantMessage(
						[{ type: "toolCall", id: "call-a", name: "lookupA", arguments: { value: "seed" } }],
						"toolUse",
					);
				} else if (sessionId.endsWith("-B") && call === 2 && serialized.includes("A_RESULT=alpha")) {
					message = createAssistantMessage(
						[
							{ type: "text", text: "B completed from A_RESULT=alpha" },
							{ type: "toolCall", id: "call-b", name: "lookupB", arguments: { value: "alpha" } },
						],
						"toolUse",
					);
				} else if (sessionId.endsWith("-C") && call === 3 && serialized.includes("B_RESULT=beta")) {
					message = createAssistantMessage(
						[
							{ type: "text", text: "C completed from B_RESULT=beta" },
							{ type: "toolCall", id: "call-c", name: "lookupC", arguments: { value: "beta" } },
						],
						"toolUse",
					);
				} else {
					message = createAssistantMessage([
						{
							type: "text",
							text: sessionId.endsWith("-C") ? "BLOCKED waiting for the next dependency" : "DEPENDENCY_NOOP",
						},
					]);
				}
				stream.push({ type: "done", reason: message.stopReason === "toolUse" ? "toolUse" : "stop", message });
			});
			return stream;
		};

		const controller = createSpeculativeSwarmController({
			enabled: true,
			model,
			agents: ["A", "B", "C", "Auditor"],
			directTakeover: false,
			maxSubagentTurns: 3,
			timeoutMs: 5_000,
			executionMode: "auto",
			toolPolicy: "readonly",
			dynamicPolicy: {
				async plan() {
					dynamicPlanCalls++;
					return ["A", "B", "C"].map((name) => ({
						title: `Action ${name}`,
						objective: `Run lookup${name} when its inputs are ready.`,
					}));
				},
				async select(input) {
					dynamicSelectCalls++;
					return {
						scores: input.drafts.map((draft) => ({
							briefId: draft.briefId,
							coherence: 1,
							utility: 1,
							novelty: 1,
							retain: true,
						})),
					};
				},
			},
		});
		expect(controller).toBeDefined();

		const userMessage: AgentMessage = {
			role: "user",
			content: [
				{
					type: "text",
					text: "First run lookup A, then use its result in lookup B, and finally use B in lookup C.",
				},
			],
			timestamp: Date.now(),
		};
		const enriched = await controller?.prepareContext({
			context: {
				systemPrompt: "Test system",
				tools: [
					createTool("lookupA", "A_RESULT=alpha"),
					createTool("lookupB", "B_RESULT=beta"),
					createTool("lookupC", "C_RESULT=gamma"),
				],
				messages: [userMessage],
			},
			model,
			thinkingLevel: "off",
			convertToLlm: identityConverter,
			streamFn,
			requestIndex: 0,
		});

		expect(enriched).toBeDefined();
		expect(dynamicPlanCalls).toBe(1);
		expect(dynamicSelectCalls).toBe(1);
		const sessionB = Array.from(seenContexts.keys()).find((sessionId) => sessionId.endsWith("-B"));
		const sessionC = Array.from(seenContexts.keys()).find((sessionId) => sessionId.endsWith("-C"));
		expect(sessionB).toBeDefined();
		expect(sessionC).toBeDefined();
		expect(seenContexts.get(sessionB ?? "")).toHaveLength(2);
		expect(seenContexts.get(sessionB ?? "")?.[1]).toContain("A_RESULT=alpha");
		expect(seenContexts.get(sessionC ?? "")).toHaveLength(3);
		expect(seenContexts.get(sessionC ?? "")?.[2]).toContain("B_RESULT=beta");

		const trace = JSON.parse(readFileSync(traceFile, "utf8").trim()) as {
			round: {
				releaseMode: string;
				dependencyTransfers: Array<{
					sourceAgent: string;
					targetAgent: string;
					result: string;
				}>;
				drafts: Array<{ agent: string; toolCalls: Array<{ toolName: string; stage: number }> }>;
			};
		};
		expect(trace.round.releaseMode).toBe("cooperative");
		expect(trace.round.dependencyTransfers).toEqual(
			expect.arrayContaining([
				expect.objectContaining({ sourceAgent: "A", targetAgent: "B", result: "A_RESULT=alpha" }),
				expect.objectContaining({ sourceAgent: "B", targetAgent: "C", result: "B_RESULT=beta" }),
			]),
		);
		expect(trace.round.drafts.find((draft) => draft.agent === "C")?.toolCalls).toContainEqual(
			expect.objectContaining({ toolName: "lookupC", stage: 3 }),
		);
	});

	it("deduplicates identical in-flight tool calls across branches", async () => {
		const model = createModel();
		let executions = 0;
		const sharedTool = createTool("lookup", "SHARED_RESULT");
		sharedTool.execute = async () => {
			executions++;
			await new Promise((resolve) => setTimeout(resolve, 10));
			return {
				content: [{ type: "text", text: "SHARED_RESULT" }],
				details: { result: "SHARED_RESULT" },
			};
		};
		const streamFn = (_model: Model<any>, _context: { messages: Message[] }, options?: { sessionId?: string }) => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = options?.sessionId?.includes("Turn0OrchestratorBrief")
					? createAssistantMessage([{ type: "text", text: "BRANCH_PROTOTYPES:\nCOLLECTIVE_DECISION_RUBRIC:" }])
					: createAssistantMessage(
							[
								{
									type: "toolCall",
									id: `call-${options?.sessionId}`,
									name: "lookup",
									arguments: { value: "same" },
								},
							],
							"toolUse",
						);
				stream.push({ type: "done", reason: message.stopReason === "toolUse" ? "toolUse" : "stop", message });
			});
			return stream;
		};
		const controller = createSpeculativeSwarmController({
			enabled: true,
			model,
			agents: ["A", "B"],
			maxSubagentTurns: 1,
			dynamicPolicy: {
				async plan() {
					return [
						{ title: "left", objective: "left lookup", toolName: "lookup", stableKey: "lookup:left" },
						{ title: "right", objective: "right lookup", toolName: "lookup", stableKey: "lookup:right" },
					];
				},
				async select(input) {
					return {
						scores: input.drafts.map((draft) => ({
							briefId: draft.briefId,
							coherence: 1,
							utility: 1,
							novelty: 1,
							retain: true,
						})),
					};
				},
			},
		});
		await controller?.prepareContext({
			context: {
				systemPrompt: "Test system",
				tools: [sharedTool],
				messages: [{ role: "user", content: [{ type: "text", text: "Run both checks." }], timestamp: Date.now() }],
			},
			model,
			thinkingLevel: "off",
			convertToLlm: identityConverter,
			streamFn,
			requestIndex: 0,
		});
		expect(executions).toBe(1);
	});

	it("ends a successful tool branch after one model stage", async () => {
		const model = createModel();
		let branchCalls = 0;
		const streamFn = (_model: Model<any>, _context: { messages: Message[] }, options?: { sessionId?: string }) => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				if (!options?.sessionId?.includes("Turn0OrchestratorBrief")) {
					branchCalls++;
				}
				const message = createAssistantMessage(
					[{ type: "toolCall", id: "call-once", name: "lookup", arguments: { value: "once" } }],
					"toolUse",
				);
				stream.push({ type: "done", reason: "toolUse", message });
			});
			return stream;
		};
		const controller = createSpeculativeSwarmController({
			enabled: true,
			model,
			agents: ["A"],
			maxSubagentTurns: 3,
			dynamicPolicy: {
				async plan() {
					return [{ id: "lookup", title: "lookup", objective: "Run lookup.", toolName: "lookup" }];
				},
				async select(input) {
					return {
						scores: input.drafts.map((draft) => ({
							briefId: draft.briefId,
							coherence: 1,
							utility: 1,
							novelty: 1,
							retain: true,
						})),
					};
				},
				async route() {
					return {
						routes: [
							{
								targetBriefId: "lookup",
								sourceBriefIds: [],
								continueOwn: true,
							},
						],
					};
				},
			},
		});
		await controller?.prepareContext({
			context: {
				systemPrompt: "Test system",
				tools: [createTool("lookup", "LOOKUP_OK")],
				messages: [{ role: "user", content: [{ type: "text", text: "Run lookup." }], timestamp: Date.now() }],
			},
			model,
			thinkingLevel: "off",
			convertToLlm: identityConverter,
			streamFn,
			requestIndex: 0,
		});
		expect(branchCalls).toBe(1);
	});

	it("executes dependency waves only after verified upstream completion", async () => {
		const model = createModel();
		const executionOrder: string[] = [];
		const streamFn = (_model: Model<any>, context: { messages: Message[] }, _options?: { sessionId?: string }) => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const serialized = JSON.stringify(context.messages);
				const isB = serialized.includes("Run B after A.");
				if (isB) {
					expect(serialized).toContain("A_RESULT=alpha");
				}
				const name = isB ? "lookupB" : "lookupA";
				executionOrder.push(name);
				const message = createAssistantMessage(
					[{ type: "toolCall", id: `call-${name}`, name, arguments: { value: name } }],
					"toolUse",
				);
				stream.push({ type: "done", reason: "toolUse", message });
			});
			return stream;
		};
		const controller = createSpeculativeSwarmController({
			enabled: true,
			model,
			agents: ["A", "B"],
			maxSubagentTurns: 1,
			dynamicPolicy: {
				async plan() {
					return [
						{ id: "a", title: "A", objective: "Run A.", toolName: "lookupA", ready: true },
						{
							id: "b",
							title: "B",
							objective: "Run B after A.",
							toolName: "lookupB",
							dependsOn: ["a"],
							ready: false,
						},
					];
				},
				async select(input) {
					return {
						scores: input.drafts.map((draft) => ({
							briefId: draft.briefId,
							coherence: 1,
							utility: 1,
							novelty: 1,
							retain: true,
						})),
					};
				},
			},
		});
		await controller?.prepareContext({
			context: {
				systemPrompt: "Test system",
				tools: [createTool("lookupA", "A_RESULT=alpha"), createTool("lookupB", "B_RESULT=beta")],
				messages: [{ role: "user", content: [{ type: "text", text: "Run A then B." }], timestamp: Date.now() }],
			},
			model,
			thinkingLevel: "off",
			convertToLlm: identityConverter,
			streamFn,
			requestIndex: 0,
		});
		expect(executionOrder).toEqual(["lookupA", "lookupB"]);
	});

	it("passes terminal failure evidence to a dependent fallback wave", async () => {
		const model = createModel();
		const executionOrder: string[] = [];
		let fallbackContext = "";
		const primary = createTool("primary", "unused");
		primary.execute = async () => {
			executionOrder.push("primary");
			throw new Error("503 service unavailable");
		};
		const fallback = createTool("fallback", "FALLBACK_OK");
		fallback.execute = async () => {
			executionOrder.push("fallback");
			return {
				content: [{ type: "text", text: "FALLBACK_OK" }],
				details: { result: "FALLBACK_OK" },
			};
		};
		const streamFn = (_model: Model<any>, context: { messages: Message[] }) => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const serialized = JSON.stringify(context.messages);
				const isFallback = serialized.includes("Run fallback after primary failure.");
				if (isFallback) {
					fallbackContext = serialized;
				}
				const name = isFallback ? "fallback" : "primary";
				const message = createAssistantMessage(
					[{ type: "toolCall", id: `call-${name}`, name, arguments: { value: "x" } }],
					"toolUse",
				);
				stream.push({ type: "done", reason: "toolUse", message });
			});
			return stream;
		};
		const controller = createSpeculativeSwarmController({
			enabled: true,
			model,
			agents: ["A"],
			maxSubagentTurns: 1,
			dynamicPolicy: {
				async plan() {
					return [
						{ id: "primary", title: "primary", objective: "Run primary.", toolName: "primary" },
						{
							id: "fallback",
							title: "fallback",
							objective: "Run fallback after primary failure.",
							toolName: "fallback",
							dependsOn: ["primary"],
							ready: false,
						},
					];
				},
				async select(input) {
					return {
						scores: input.drafts.map((draft) => ({
							briefId: draft.briefId,
							coherence: 1,
							utility: 1,
							novelty: 1,
							retain: true,
						})),
					};
				},
			},
		});
		await controller?.prepareContext({
			context: {
				systemPrompt: "Test system",
				tools: [primary, fallback],
				messages: [{ role: "user", content: [{ type: "text", text: "Try primary then fallback." }], timestamp: Date.now() }],
			},
			model,
			thinkingLevel: "off",
			convertToLlm: identityConverter,
			streamFn,
			requestIndex: 0,
		});
		expect(executionOrder).toEqual(["primary", "fallback"]);
		expect(fallbackContext).toContain("503 service unavailable");
	});

	it("uses learned wave stages and sizes instead of launching every branch at once", async () => {
		const model = createModel();
		const selectedDraftCounts: number[] = [];
		const streamFn = (_model: Model<any>, context: { messages: Message[] }) => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const serialized = JSON.stringify(context.messages);
				const match = serialized.match(/Run tool([A-D])/);
				const toolName = `tool${match?.[1] ?? "A"}`;
				const message = createAssistantMessage(
					[{ type: "toolCall", id: `call-${toolName}`, name: toolName, arguments: {} }],
					"toolUse",
				);
				stream.push({ type: "done", reason: "toolUse", message });
			});
			return stream;
		};
		const controller = createSpeculativeSwarmController({
			enabled: true,
			model,
			agents: ["A", "B", "C", "D"],
			maxSubagentTurns: 1,
			dynamicPolicy: {
				async plan() {
					return ["A", "B", "C", "D"].map((suffix, index) => ({
						id: `tool-${suffix.toLowerCase()}`,
						title: `tool${suffix}`,
						objective: `Run tool${suffix}.`,
						toolName: `tool${suffix}`,
						executionWave: index < 2 ? 1 : 2,
						recommendedWaveSize: 2,
						laterWaveSize: 2,
					}));
				},
				async select(input) {
					selectedDraftCounts.push(input.drafts.length);
					return {
						scores: input.drafts.map((draft) => ({
							briefId: draft.briefId,
							coherence: 1,
							utility: 1,
							novelty: 1,
							retain: true,
						})),
					};
				},
			},
		});
		await controller?.prepareContext({
			context: {
				systemPrompt: "Test system",
				tools: ["A", "B", "C", "D"].map((suffix) => createTool(`tool${suffix}`, `${suffix}_OK`)),
				messages: [{ role: "user", content: [{ type: "text", text: "Run all tools." }], timestamp: Date.now() }],
			},
			model,
			thinkingLevel: "off",
			convertToLlm: identityConverter,
			streamFn,
			requestIndex: 0,
		});
		expect(selectedDraftCounts).toEqual([2, 4]);
	});

	it("does not re-run the dynamic swarm for a main-agent tool result", async () => {
		const model = createModel();
		let planCalls = 0;
		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "complete" }]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};
		const controller = createSpeculativeSwarmController({
			enabled: true,
			model,
			agents: ["A"],
			dynamicPolicy: {
				async plan() {
					planCalls++;
					return [{ id: "answer", title: "answer", objective: "Answer." }];
				},
				async select(input) {
					return {
						scores: input.drafts.map((draft) => ({
							briefId: draft.briefId,
							coherence: 1,
							utility: 1,
							novelty: 1,
							retain: true,
						})),
					};
				},
			},
		});
		const user: AgentMessage = {
			role: "user",
			content: [{ type: "text", text: "Start." }],
			timestamp: Date.now(),
		};
		await controller?.prepareContext({
			context: { systemPrompt: "Test system", tools: [], messages: [user] },
			model,
			thinkingLevel: "off",
			convertToLlm: identityConverter,
			streamFn,
			requestIndex: 0,
		});
		const toolResult = {
			role: "toolResult",
			toolCallId: "main-call",
			toolName: "lookup",
			content: [{ type: "text", text: "MAIN_RESULT" }],
			isError: false,
			timestamp: Date.now(),
		} as AgentMessage;
		const second = await controller?.prepareContext({
			context: { systemPrompt: "Test system", tools: [], messages: [user, toolResult] },
			model,
			thinkingLevel: "off",
			convertToLlm: identityConverter,
			streamFn,
			requestIndex: 1,
		});
		expect(second).toBeUndefined();
		expect(planCalls).toBe(1);
	});

	it("returns a coherent completed branch directly as the main response", async () => {
		const model = createModel();
		const streamFn = (_model: Model<any>, _context: { messages: Message[] }, options?: { sessionId?: string }) => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([
					{
						type: "text",
						text: options?.sessionId?.includes("Turn0OrchestratorBrief")
							? "BRANCH_PROTOTYPES:\nCOLLECTIVE_DECISION_RUBRIC:"
							: "Verified final answer from the coherent branch.",
					},
				]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};
		const controller = createSpeculativeSwarmController({
			enabled: true,
			model,
			agents: ["A"],
			takeoverThreshold: 0.8,
			dynamicPolicy: {
				async plan() {
					return [{ title: "answer", objective: "Produce the verified answer.", stableKey: "answer" }];
				},
				async select(input) {
					return {
						scores: input.drafts.map((draft) => ({
							briefId: draft.briefId,
							coherence: 1,
							utility: 1,
							novelty: 1,
							retain: true,
						})),
						taskSolvable: true,
						taskStatus: "complete",
						mainDecision: "takeover",
						mainBriefIds: [input.drafts[0].briefId],
						mainConfidence: 0.95,
					};
				},
			},
		});
		const prepared = await controller?.prepareTurn?.({
			context: {
				systemPrompt: "Test system",
				tools: [],
				messages: [{ role: "user", content: [{ type: "text", text: "Answer directly." }], timestamp: Date.now() }],
			},
			model,
			thinkingLevel: "off",
			convertToLlm: identityConverter,
			streamFn,
			requestIndex: 0,
		});
		expect(prepared?.context).toBeUndefined();
		expect(prepared?.response?.content).toEqual([
			{ type: "text", text: "Verified final answer from the coherent branch." },
		]);
	});

	it("takes over structurally when every planned tool action succeeded", async () => {
		const model = createModel();
		const streamFn = (_model: Model<any>, context: { messages: Message[] }) => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const serialized = JSON.stringify(context.messages);
				const name = serialized.includes("Run beta.") ? "beta" : "alpha";
				const message = createAssistantMessage(
					[{ type: "toolCall", id: `call-${name}`, name, arguments: {} }],
					"toolUse",
				);
				stream.push({ type: "done", reason: "toolUse", message });
			});
			return stream;
		};
		const controller = createSpeculativeSwarmController({
			enabled: true,
			model,
			agents: ["A", "B"],
			takeoverThreshold: 0.8,
			dynamicPolicy: {
				async plan() {
					return [
						{ id: "alpha", title: "alpha", objective: "Run alpha.", toolName: "alpha" },
						{ id: "beta", title: "beta", objective: "Run beta.", toolName: "beta" },
					];
				},
				async select(input) {
					return {
						scores: input.drafts.map((draft) => ({
							briefId: draft.briefId,
							coherence: 1,
							utility: 1,
							novelty: 1,
							retain: true,
						})),
						taskComplete: false,
						mainDecision: "delegate",
						mainConfidence: 0,
					};
				},
			},
		});
		const prepared = await controller?.prepareTurn?.({
			context: {
				systemPrompt: "Test system",
				tools: [createTool("alpha", "ALPHA_OK"), createTool("beta", "BETA_OK")],
				messages: [{ role: "user", content: [{ type: "text", text: "Run alpha and beta." }], timestamp: Date.now() }],
			},
			model,
			thinkingLevel: "off",
			convertToLlm: identityConverter,
			streamFn,
			requestIndex: 0,
		});
		expect(prepared?.context).toBeUndefined();
		expect(prepared?.response?.content[0]).toEqual(
			expect.objectContaining({
				type: "text",
				text: expect.stringContaining("ALPHA_OK"),
			}),
		);
		expect(prepared?.response?.content[0]).toEqual(
			expect.objectContaining({
				type: "text",
				text: expect.stringContaining("BETA_OK"),
			}),
		);
	});

	it("returns a completed truthful failure report without another main-model generation", async () => {
		const model = createModel();
		let generationCalls = 0;
		const streamFn = () => {
			generationCalls++;
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "The assigned action failed." }]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};
		const controller = createSpeculativeSwarmController({
			enabled: true,
			model,
			agents: ["A"],
			dynamicPolicy: {
				async plan() {
					return [{ id: "failed", title: "failed", objective: "Attempt the required action." }];
				},
				async select(input) {
					return {
						scores: input.drafts.map((draft) => ({
							briefId: draft.briefId,
							coherence: 1,
							utility: 1,
							novelty: 1,
							retain: true,
						})),
						taskSolvable: false,
						taskComplete: true,
						taskStatus: "complete",
						mainDecision: "takeover",
						mainBriefIds: [input.drafts[0].briefId],
						mainConfidence: 0.95,
						mustReportFailure: true,
						failureReports: [
							{
								briefId: input.drafts[0].briefId,
								failureType: "invalid_call",
								message: "The required action failed.",
							},
						],
					};
				},
			},
		});
		const prepared = await controller?.prepareTurn?.({
			context: {
				systemPrompt: "Test system",
				tools: [],
				messages: [{ role: "user", content: [{ type: "text", text: "Run the action." }], timestamp: Date.now() }],
			},
			model,
			thinkingLevel: "off",
			convertToLlm: identityConverter,
			streamFn,
			requestIndex: 0,
		});
		expect(generationCalls).toBe(1);
		expect(prepared?.context).toBeUndefined();
		expect(prepared?.response?.content[0]).toEqual(
			expect.objectContaining({
				type: "text",
				text: expect.stringContaining("TASK_STATUS: COMPLETE_WITH_REPORTED_FAILURE"),
			}),
		);
	});

	it("does not force takeover when the current plan is unsolvable but incomplete", async () => {
		const model = createModel();
		let generationCalls = 0;
		const streamFn = () => {
			generationCalls++;
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "The action failed before completion." }]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};
		const controller = createSpeculativeSwarmController({
			enabled: true,
			model,
			agents: ["A"],
			dynamicPolicy: {
				async plan() {
					return [{ id: "failed", title: "failed", objective: "Attempt the required action." }];
				},
				async select(input) {
					return {
						scores: input.drafts.map((draft) => ({
							briefId: draft.briefId,
							coherence: 0,
							utility: 0,
							novelty: 0,
							retain: false,
						})),
						taskSolvable: false,
						taskComplete: false,
						taskStatus: "unsolvable_current_plan",
						mainDecision: "takeover",
						mainBriefIds: [input.drafts[0].briefId],
						mainConfidence: 1,
						mustReportFailure: true,
					};
				},
			},
		});
		const prepared = await controller?.prepareTurn?.({
			context: {
				systemPrompt: "Test system",
				tools: [],
				messages: [{ role: "user", content: [{ type: "text", text: "Run the action." }], timestamp: Date.now() }],
			},
			model,
			thinkingLevel: "off",
			convertToLlm: identityConverter,
			streamFn,
			requestIndex: 0,
		});
		expect(generationCalls).toBe(1);
		expect(prepared).toBeUndefined();
	});

	it("does not retry terminal provider failures even when routing requests repair", async () => {
		const model = createModel();
		let branchCalls = 0;
		const failingTool = createTool("unstable", "unused");
		failingTool.execute = async () => {
			throw new Error("503 service unavailable");
		};
		const streamFn = (_model: Model<any>, _context: { messages: Message[] }, _options?: { sessionId?: string }) => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				branchCalls++;
				const message = createAssistantMessage(
					[
						{
							type: "toolCall",
							id: `failure-${branchCalls}`,
							name: "unstable",
							arguments: { value: `attempt-${branchCalls}` },
						},
					],
					"toolUse",
				);
				stream.push({ type: "done", reason: "toolUse", message });
			});
			return stream;
		};
		const controller = createSpeculativeSwarmController({
			enabled: true,
			model,
			agents: ["A"],
			maxSubagentTurns: 4,
			dynamicPolicy: {
				async plan() {
					return [{ id: "unstable", title: "unstable", objective: "Call unstable.", toolName: "unstable" }];
				},
				async select(input) {
					return {
						scores: input.drafts.map((draft) => ({
							briefId: draft.briefId,
							coherence: 0,
							utility: 0,
							novelty: 0,
							retain: false,
						})),
						taskSolvable: false,
						taskStatus: "unsolvable_current_plan",
					};
				},
				async route(input) {
					return {
						routes: [
							{
								targetBriefId: input.actions[0].id ?? "unstable",
								sourceBriefIds: [],
								retryOwnErrors: true,
							},
						],
					};
				},
			},
		});
		await controller?.prepareTurn?.({
			context: {
				systemPrompt: "Test system",
				tools: [failingTool],
				messages: [{ role: "user", content: [{ type: "text", text: "Call unstable." }], timestamp: Date.now() }],
			},
			model,
			thinkingLevel: "off",
			convertToLlm: identityConverter,
			streamFn,
			requestIndex: 0,
		});
		expect(branchCalls).toBe(1);
	});
});
