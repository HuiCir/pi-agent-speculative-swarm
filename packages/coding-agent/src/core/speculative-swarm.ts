import { appendFileSync } from "node:fs";
import {
	Agent,
	type AgentContext,
	type AgentEvent,
	type AgentMessage,
	type AgentTool,
	type SpeculativeSwarmController,
	type SpeculativeSwarmPrepareContext,
	type ThinkingLevel,
} from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Model, TextContent, ToolResultMessage } from "@earendil-works/pi-ai";

const SWARM_MEMORY_MARKER = "<residual_swarm_memory";
const ORCHESTRATOR_BRIEF_AGENT = "Turn0OrchestratorBrief";
const DEFAULT_AGENTS = ["SwarmAgent-A", "SwarmAgent-B", "SwarmAgent-C", "SwarmAgent-D"];
const MUTATING_TOOL_NAMES = new Set(["edit", "write"]);

export type SpeculativeSwarmToolPolicy = "none" | "readonly" | "all";

export interface SpeculativeSwarmRuntimeOptions {
	enabled?: boolean;
	model: Model<any>;
	thinkingLevel?: ThinkingLevel;
	agents?: string[];
	rounds?: number;
	maxSubagentTurns?: number;
	timeoutMs?: number;
	toolPolicy?: SpeculativeSwarmToolPolicy;
	allowBash?: boolean;
	maxDraftChars?: number;
	maxMemoryRounds?: number;
	twoStage?: boolean;
}

interface SwarmToolCallSummary {
	agent: string;
	toolName: string;
	args: unknown;
	isError?: boolean;
	resultPreview?: string;
}

interface BranchBrief {
	id: string;
	title: string;
	objective: string;
	whyDistinct: string;
	suggestedChecks: string[];
	risk: string;
}

interface SwarmDraft {
	agent: string;
	roundIndex: number;
	briefId: string;
	branchTitle: string;
	latencyMs: number;
	turns: number;
	status: "ok" | "error" | "aborted" | "timeout";
	finalText: string;
	toolCalls: SwarmToolCallSummary[];
	score: number;
	usage?: AssistantMessage["usage"];
	clusterId?: string;
	eliminationReason?: string;
	errorMessage?: string;
}

interface OrchestratorBriefDraft {
	agent: string;
	roundIndex: number;
	latencyMs: number;
	status: "ok" | "error" | "aborted" | "timeout";
	finalText: string;
	briefs: BranchBrief[];
	usage?: AssistantMessage["usage"];
	errorMessage?: string;
}

interface SwarmRound {
	roundIndex: number;
	requestIndex: number;
	startedAt: number;
	latencyMs: number;
	releaseMode: "early" | "barrier" | "background";
	backgroundComplete: boolean;
	drafts: SwarmDraft[];
	preflight: OrchestratorBriefDraft;
	activeBriefs: BranchBrief[];
	retainedDrafts: SwarmDraft[];
	suppressedDrafts: SwarmDraft[];
	selection: string;
}

interface CollectiveDecision {
	selection: string;
	scoredDrafts: SwarmDraft[];
	retainedDrafts: SwarmDraft[];
	suppressedDrafts: SwarmDraft[];
	nextBriefs: BranchBrief[];
}

interface BranchRuntimeState {
	brief: BranchBrief;
	lastAgent?: string;
	lastDraft?: SwarmDraft;
	completedRounds: number;
	status: "active" | "suppressed";
	updatedAt: number;
}

interface TrackedPromise<T> {
	promise: Promise<T>;
	settled: boolean;
	value?: T;
	error?: unknown;
}

export function createSpeculativeSwarmController(
	options: SpeculativeSwarmRuntimeOptions,
): SpeculativeSwarmController | undefined {
	if (!options.enabled) {
		return undefined;
	}
	return new ResidualSpeculativeSwarmController(options);
}

export function isSpeculativeSwarmMemoryMessage(message: AgentMessage): boolean {
	if (message.role !== "user") {
		return false;
	}
	return getText(message).includes(SWARM_MEMORY_MARKER);
}

class ResidualSpeculativeSwarmController implements SpeculativeSwarmController {
	private readonly options: Required<
		Pick<
			SpeculativeSwarmRuntimeOptions,
			| "agents"
			| "rounds"
			| "maxSubagentTurns"
			| "timeoutMs"
			| "toolPolicy"
			| "allowBash"
			| "maxDraftChars"
			| "maxMemoryRounds"
			| "twoStage"
		>
	> &
		Pick<SpeculativeSwarmRuntimeOptions, "model" | "thinkingLevel">;
	private memory: SwarmRound[] = [];
	private activeBriefs: BranchBrief[] = [];
	private branchRuntime = new Map<string, BranchRuntimeState>();
	private branchCounter = 0;
	private activePrepare?: Promise<AgentContext | undefined>;

	constructor(options: SpeculativeSwarmRuntimeOptions) {
		this.options = {
			model: options.model,
			thinkingLevel: options.thinkingLevel,
			agents: options.agents?.length ? options.agents : DEFAULT_AGENTS,
			rounds: Math.max(1, options.rounds ?? 1),
			maxSubagentTurns: Math.max(1, options.maxSubagentTurns ?? 3),
			timeoutMs: Math.max(1_000, options.timeoutMs ?? 120_000),
			toolPolicy: options.toolPolicy ?? "readonly",
			allowBash: options.allowBash ?? false,
			maxDraftChars: Math.max(1_000, options.maxDraftChars ?? 16_000),
			maxMemoryRounds: Math.max(1, options.maxMemoryRounds ?? 6),
			twoStage: options.twoStage ?? true,
		};
	}

	async prepareContext(input: SpeculativeSwarmPrepareContext): Promise<AgentContext | undefined> {
		if (input.signal?.aborted) {
			return undefined;
		}

		if (this.activePrepare) {
			return this.activePrepare;
		}

		this.activePrepare = this.prepareContextOnce(input).finally(() => {
			this.activePrepare = undefined;
		});
		return this.activePrepare;
	}

	private async prepareContextOnce(input: SpeculativeSwarmPrepareContext): Promise<AgentContext | undefined> {
		const cleanMessages = input.context.messages.filter((message) => !isSpeculativeSwarmMemoryMessage(message));
		const lastMessage = cleanMessages[cleanMessages.length - 1];
		if (!lastMessage || (lastMessage.role !== "user" && lastMessage.role !== "toolResult")) {
			return undefined;
		}

		let latestDrafts: SwarmDraft[] = [];
		let latestSelection = "";
		const startedAt = Date.now();

		for (let i = 0; i < this.options.rounds; i++) {
			if (input.signal?.aborted) {
				break;
			}

			const round = await this.runSwarmRoundFifo(cleanMessages, input);
			if (!round) {
				break;
			}
			latestDrafts = round.drafts;
			latestSelection = round.selection;
		}

		if (latestDrafts.length === 0) {
			return undefined;
		}

		const memoryMessage = this.buildMemoryMessage({
			requestIndex: input.requestIndex,
			mainModel: input.model,
			mainThinkingLevel: input.thinkingLevel,
			latencyMs: Date.now() - startedAt,
			latestDrafts,
			latestSelection,
		});

		return {
			...input.context,
			messages: [
				...cleanMessages,
				{
					role: "user",
					content: [{ type: "text", text: memoryMessage }],
					timestamp: Date.now(),
				},
			],
		};
	}

	private async runSubagent(
		agentName: string,
		roundIndex: number,
		baseMessages: AgentMessage[],
		input: SpeculativeSwarmPrepareContext,
		brief: BranchBrief,
	): Promise<SwarmDraft> {
		const startedAt = Date.now();
		const abortController = new AbortController();
		const onAbort = () => abortController.abort();
		input.signal?.addEventListener("abort", onAbort, { once: true });

		let turns = 0;
		const toolCalls = new Map<string, SwarmToolCallSummary>();

		const agent = new Agent({
			initialState: {
				systemPrompt: this.buildSubagentSystemPrompt(agentName, input.context.systemPrompt),
				model: this.options.model,
				thinkingLevel: this.options.thinkingLevel ?? "off",
				tools: this.filterTools(input.context.tools ?? []),
				messages: [
					...baseMessages,
					{
						role: "user",
						content: [{ type: "text", text: this.buildSubagentRuntimeMessage(roundIndex, brief) }],
						timestamp: Date.now(),
					},
				],
			},
			convertToLlm: input.convertToLlm,
			streamFn: input.streamFn,
			toolExecution: "parallel",
			sessionId: `swarm-branch-${brief.id}-${agentName}`,
			transport: "auto",
		});

		agent.subscribe((event) => {
			this.captureSubagentEvent(agentName, event, toolCalls);
			if (event.type === "turn_end") {
				turns++;
				if (turns >= this.options.maxSubagentTurns) {
					agent.abort();
				}
			}
		});

		const timedOut = await this.runWithTimeout(agent.continue(), abortController, this.options.timeoutMs);
		input.signal?.removeEventListener("abort", onAbort);

		const finalAssistant = findLastAssistant(agent.state.messages);
		const finalText = finalAssistant ? getText(finalAssistant) : "";
		const status = timedOut
			? "timeout"
			: abortController.signal.aborted
				? "aborted"
				: finalAssistant?.stopReason === "error"
					? "error"
					: "ok";

		return {
			agent: agentName,
			roundIndex,
			briefId: brief.id,
			branchTitle: brief.title,
			latencyMs: Date.now() - startedAt,
			turns,
			status,
			finalText: truncateText(finalText, this.options.maxDraftChars),
			toolCalls: Array.from(toolCalls.values()),
			score: 0,
			usage: finalAssistant?.usage,
			errorMessage: finalAssistant?.errorMessage,
		};
	}

	private async runSwarmRoundFifo(
		cleanMessages: AgentMessage[],
		input: SpeculativeSwarmPrepareContext,
	): Promise<SwarmRound | undefined> {
		const roundIndex = this.memory.length + 1;
		const roundStartedAt = Date.now();
		const activeBriefs = this.prepareBranchBriefs(cleanMessages, input.requestIndex, this.options.agents.length);
		for (const brief of activeBriefs) {
			this.ensureBranchRuntime(brief);
		}

		const shouldRunPreflight = this.shouldRunPreflight(input.requestIndex);
		const preflightTask = trackPromise(
			shouldRunPreflight
				? this.runOrchestratorBrief(roundIndex, cleanMessages, input, activeBriefs)
				: Promise.resolve(this.createSkippedPreflight(roundIndex, activeBriefs)),
		);
		const draftTasks = this.options.agents.map((agentName, index) =>
			trackPromise(this.runSubagent(agentName, roundIndex, cleanMessages, input, activeBriefs[index])),
		);

		const replayDraft = this.createRuntimeReplayDraft(roundIndex, activeBriefs);
		const earlyDrafts = replayDraft
			? [replayDraft]
			: await this.waitForEarlyDraft(draftTasks, activeBriefs, cleanMessages);
		if (earlyDrafts.length === 0) {
			return undefined;
		}

		const earlyPreflight = preflightTask.settled
			? (preflightTask.value ?? this.createPlaceholderPreflight(roundIndex, activeBriefs, roundStartedAt))
			: this.createPlaceholderPreflight(roundIndex, activeBriefs, roundStartedAt);
		const earlyDecision = this.buildCollectiveDecision(earlyDrafts, activeBriefs, earlyPreflight, cleanMessages);
		const releaseMode = draftTasks.every((task) => task.settled) && preflightTask.settled ? "barrier" : "early";
		const earlyRound: SwarmRound = {
			roundIndex,
			requestIndex: input.requestIndex,
			startedAt: roundStartedAt,
			latencyMs: Date.now() - roundStartedAt,
			releaseMode,
			backgroundComplete: releaseMode === "barrier",
			drafts: earlyDecision.scoredDrafts,
			preflight: earlyPreflight,
			activeBriefs,
			retainedDrafts: earlyDecision.retainedDrafts,
			suppressedDrafts: earlyDecision.suppressedDrafts,
			selection: earlyDecision.selection,
		};
		this.upsertRound(earlyRound);
		this.applyDecisionToRuntime(earlyDecision, activeBriefs);
		this.traceRound(earlyRound);

		if (releaseMode === "early") {
			void this.finalizeBackgroundRound({
				roundIndex,
				requestIndex: input.requestIndex,
				roundStartedAt,
				activeBriefs,
				preflightTask,
				draftTasks,
				cleanMessages,
			});
		}

		return earlyRound;
	}

	private createRuntimeReplayDraft(roundIndex: number, activeBriefs: BranchBrief[]): SwarmDraft | undefined {
		if (process.env.PI_SWARM_ENABLE_RUNTIME_REPLAY !== "1") {
			return undefined;
		}
		if (this.memory.length === 0) {
			return undefined;
		}
		for (const brief of activeBriefs) {
			const state = this.branchRuntime.get(brief.id);
			const draft = state?.lastDraft;
			if (!draft || state.status !== "active") {
				continue;
			}
			if (draft.score < 4 || hasUnsupportedAbsenceClaim(draft)) {
				continue;
			}
			if (normalizeText(draft.finalText).length < 160 && draft.toolCalls.length === 0) {
				continue;
			}
			return {
				...draft,
				agent: `RuntimeReplay:${state.lastAgent ?? draft.agent}`,
				roundIndex,
				latencyMs: 0,
				turns: 0,
				score: Math.max(1, draft.score - 0.25),
			};
		}
		return undefined;
	}

	private shouldRunPreflight(requestIndex: number): boolean {
		return requestIndex === 0 && this.memory.length === 0;
	}

	private async waitForEarlyDraft(
		draftTasks: TrackedPromise<SwarmDraft>[],
		activeBriefs: BranchBrief[],
		baseMessages: AgentMessage[],
	): Promise<SwarmDraft[]> {
		const pending = new Set(draftTasks.map((_, index) => index));
		const completed: SwarmDraft[] = [];
		while (pending.size > 0) {
			const result = await Promise.race(
				Array.from(pending).map((index) =>
					draftTasks[index].promise
						.then((draft) => ({ index, draft }))
						.catch((error) => ({ index, draft: this.createErrorDraft(index, activeBriefs, error) })),
				),
			);
			pending.delete(result.index);
			const brief = activeBriefs[result.index];
			const draft = {
				...result.draft,
				score: this.scoreDraft(result.draft, brief, baseMessages),
			};
			completed.push(draft);
			this.updateBranchRuntime(brief, draft, "active");
			if (this.isSatisfactoryEarlyDraft(draft)) {
				return completed;
			}
		}
		return completed;
	}

	private async finalizeBackgroundRound(input: {
		roundIndex: number;
		requestIndex: number;
		roundStartedAt: number;
		activeBriefs: BranchBrief[];
		preflightTask: TrackedPromise<OrchestratorBriefDraft>;
		draftTasks: TrackedPromise<SwarmDraft>[];
		cleanMessages: AgentMessage[];
	}): Promise<void> {
		const drafts = await Promise.all(
			input.draftTasks.map((task, index) =>
				task.promise.catch((error) => this.createErrorDraft(index, input.activeBriefs, error)),
			),
		);
		const preflight = await input.preflightTask.promise.catch(() =>
			this.createPlaceholderPreflight(input.roundIndex, input.activeBriefs, input.roundStartedAt),
		);
		const decision = this.buildCollectiveDecision(drafts, input.activeBriefs, preflight, input.cleanMessages);
		const round: SwarmRound = {
			roundIndex: input.roundIndex,
			requestIndex: input.requestIndex,
			startedAt: input.roundStartedAt,
			latencyMs: Date.now() - input.roundStartedAt,
			releaseMode: "background",
			backgroundComplete: true,
			drafts: decision.scoredDrafts,
			preflight,
			activeBriefs: input.activeBriefs,
			retainedDrafts: decision.retainedDrafts,
			suppressedDrafts: decision.suppressedDrafts,
			selection: decision.selection,
		};
		this.upsertRound(round);
		this.applyDecisionToRuntime(decision, input.activeBriefs);
		this.traceRound(round);
	}

	private isSatisfactoryEarlyDraft(draft: SwarmDraft): boolean {
		const successfulTools = draft.toolCalls.filter((call) => !call.isError).length;
		const textLength = normalizeText(draft.finalText).length;
		return (
			draft.status === "ok" &&
			draft.score >= 3 &&
			(successfulTools > 0 || textLength > 160) &&
			!hasUnsupportedAbsenceClaim(draft)
		);
	}

	private createErrorDraft(index: number, activeBriefs: BranchBrief[], error: unknown): SwarmDraft {
		const brief = activeBriefs[index];
		return {
			agent: this.options.agents[index] ?? `SwarmAgent-${index + 1}`,
			roundIndex: this.memory.length + 1,
			briefId: brief.id,
			branchTitle: brief.title,
			latencyMs: 0,
			turns: 0,
			status: "error",
			finalText: "",
			toolCalls: [],
			score: 0,
			errorMessage: error instanceof Error ? error.message : String(error),
		};
	}

	private createPlaceholderPreflight(
		roundIndex: number,
		briefs: BranchBrief[],
		roundStartedAt: number,
	): OrchestratorBriefDraft {
		return {
			agent: ORCHESTRATOR_BRIEF_AGENT,
			roundIndex,
			latencyMs: Date.now() - roundStartedAt,
			status: "aborted",
			finalText:
				"FIFO early release: Turn0 preflight was still running. Treat this as no voting rubric yet; use the fastest satisfactory draft as provisional evidence.",
			briefs,
		};
	}

	private createSkippedPreflight(roundIndex: number, briefs: BranchBrief[]): OrchestratorBriefDraft {
		return {
			agent: ORCHESTRATOR_BRIEF_AGENT,
			roundIndex,
			latencyMs: 0,
			status: "aborted",
			finalText:
				"Turn0 preflight skipped after initial branch creation. Use retained branch runtime state and FIFO draft evidence.",
			briefs,
		};
	}

	private upsertRound(round: SwarmRound): void {
		const existingIndex = this.memory.findIndex((item) => item.roundIndex === round.roundIndex);
		if (existingIndex >= 0) {
			this.memory[existingIndex] = round;
		} else {
			this.memory.push(round);
		}
		this.memory = this.memory.slice(-this.options.maxMemoryRounds);
	}

	private ensureBranchRuntime(brief: BranchBrief): void {
		if (this.branchRuntime.has(brief.id)) {
			return;
		}
		this.branchRuntime.set(brief.id, {
			brief,
			completedRounds: 0,
			status: "active",
			updatedAt: Date.now(),
		});
	}

	private updateBranchRuntime(brief: BranchBrief, draft: SwarmDraft, status: BranchRuntimeState["status"]): void {
		if (draft.agent.startsWith("RuntimeReplay:")) {
			return;
		}
		const existing = this.branchRuntime.get(brief.id);
		this.branchRuntime.set(brief.id, {
			brief,
			lastAgent: draft.agent,
			lastDraft: draft,
			completedRounds: (existing?.completedRounds ?? 0) + 1,
			status,
			updatedAt: Date.now(),
		});
	}

	private applyDecisionToRuntime(decision: CollectiveDecision, briefs: BranchBrief[]): void {
		const briefById = new Map(briefs.map((brief) => [brief.id, brief]));
		for (const draft of decision.retainedDrafts) {
			const brief = briefById.get(draft.briefId);
			if (brief) {
				this.updateBranchRuntime(brief, draft, "active");
			}
		}
		for (const draft of decision.suppressedDrafts) {
			const brief = briefById.get(draft.briefId);
			if (brief) {
				this.updateBranchRuntime(brief, draft, "suppressed");
			}
		}
		this.activeBriefs = decision.nextBriefs.slice(0, this.options.agents.length);
	}

	private async runOrchestratorBrief(
		roundIndex: number,
		baseMessages: AgentMessage[],
		input: SpeculativeSwarmPrepareContext,
		briefs: BranchBrief[],
	): Promise<OrchestratorBriefDraft> {
		const startedAt = Date.now();
		const abortController = new AbortController();
		const onAbort = () => abortController.abort();
		input.signal?.addEventListener("abort", onAbort, { once: true });

		const agent = new Agent({
			initialState: {
				systemPrompt: this.buildOrchestratorBriefPrompt(roundIndex, briefs, input.context.systemPrompt),
				model: this.options.model,
				thinkingLevel: this.options.thinkingLevel ?? "off",
				tools: [],
				messages: baseMessages.slice(),
			},
			convertToLlm: input.convertToLlm,
			streamFn: input.streamFn,
			toolExecution: "parallel",
			sessionId: `swarm-${ORCHESTRATOR_BRIEF_AGENT}-${roundIndex}-${input.requestIndex}`,
			transport: "auto",
		});

		let turns = 0;
		agent.subscribe((event) => {
			if (event.type === "turn_end") {
				turns++;
				if (turns >= 1) {
					agent.abort();
				}
			}
		});

		const timedOut = await this.runWithTimeout(
			agent.continue(),
			abortController,
			Math.min(this.options.timeoutMs, 30_000),
		);
		input.signal?.removeEventListener("abort", onAbort);

		const finalAssistant = findLastAssistant(agent.state.messages);
		const finalText = finalAssistant ? getText(finalAssistant) : "";
		const status = timedOut
			? "timeout"
			: abortController.signal.aborted
				? "aborted"
				: finalAssistant?.stopReason === "error"
					? "error"
					: "ok";

		return {
			agent: ORCHESTRATOR_BRIEF_AGENT,
			roundIndex,
			latencyMs: Date.now() - startedAt,
			status,
			finalText: truncateText(finalText, this.options.maxDraftChars),
			briefs,
			usage: finalAssistant?.usage,
			errorMessage: finalAssistant?.errorMessage,
		};
	}

	private async runWithTimeout(
		promise: Promise<void>,
		abortController: AbortController,
		timeoutMs: number,
	): Promise<boolean> {
		let timeout: NodeJS.Timeout | undefined;
		const timeoutPromise = new Promise<"timeout">((resolve) => {
			timeout = setTimeout(() => {
				abortController.abort();
				resolve("timeout");
			}, timeoutMs);
		});

		const result = await Promise.race([promise.then(() => "done" as const), timeoutPromise]);
		if (timeout) {
			clearTimeout(timeout);
		}
		if (result === "timeout") {
			await promise.catch(() => undefined);
			return true;
		}
		return false;
	}

	private captureSubagentEvent(
		agentName: string,
		event: AgentEvent,
		toolCalls: Map<string, SwarmToolCallSummary>,
	): void {
		if (event.type === "tool_execution_start") {
			toolCalls.set(event.toolCallId, {
				agent: agentName,
				toolName: event.toolName,
				args: event.args,
			});
			return;
		}
		if (event.type === "tool_execution_end") {
			const existing = toolCalls.get(event.toolCallId);
			toolCalls.set(event.toolCallId, {
				agent: agentName,
				toolName: event.toolName,
				args: existing?.args,
				isError: event.isError,
				resultPreview: summarizeToolResult(event.result),
			});
		}
	}

	private filterTools(tools: AgentTool<any>[]): AgentTool<any>[] {
		if (this.options.toolPolicy === "none") {
			return [];
		}
		if (this.options.toolPolicy === "all") {
			return tools.slice();
		}
		return tools.filter((tool) => {
			if (MUTATING_TOOL_NAMES.has(tool.name)) {
				return false;
			}
			if (tool.name === "bash" && !this.options.allowBash) {
				return false;
			}
			return true;
		});
	}

	private prepareBranchBriefs(baseMessages: AgentMessage[], requestIndex: number, count: number): BranchBrief[] {
		const retained = this.activeBriefs.slice(0, count);
		const needed = Math.max(0, count - retained.length);
		if (needed === 0) {
			return retained;
		}
		return [...retained, ...this.createFreshBranchBriefs(baseMessages, requestIndex, needed)];
	}

	private createFreshBranchBriefs(baseMessages: AgentMessage[], requestIndex: number, count: number): BranchBrief[] {
		const userText = truncateText(
			getText(baseMessages[baseMessages.length - 1] ?? ({ role: "user" } as AgentMessage)),
			900,
		);
		const templates = [
			{
				title: "Direct evidence path",
				objective:
					"Follow the most explicit task target and gather the minimum direct evidence needed for a usable draft.",
				whyDistinct: "Optimizes for speed and concrete evidence instead of broad exploration.",
				suggestedChecks: [
					"Identify the concrete deliverable",
					"Call only the most direct read/API tools",
					"Record exact ids, paths, parameters, or returned values",
				],
				risk: "May miss hidden dependencies or later-stage constraints.",
			},
			{
				title: "Dependency graph path",
				objective:
					"Infer prerequisite steps, blocked nodes, and information that must be staged before the final action can be correct.",
				whyDistinct: "Optimizes for long-horizon ordering and residual reuse across future rounds.",
				suggestedChecks: [
					"List prerequisites and locks",
					"Separate known, missing, and blocked facts",
					"Find partial work reusable by later turns",
				],
				risk: "May spend too much budget on planning if the task is actually shallow.",
			},
			{
				title: "Parameter and entity path",
				objective:
					"Resolve ambiguous entities, parameter values, schema constraints, and mismatch risks before execution.",
				whyDistinct: "Optimizes for preventing entity leakage and incorrect tool arguments.",
				suggestedChecks: [
					"Extract all candidate entities and parameters",
					"Verify exact names and schemas",
					"Flag mismatches that would invalidate the task",
				],
				risk: "May over-focus on parameters and under-cover the final synthesis.",
			},
			{
				title: "Counterfactual validation path",
				objective:
					"Look for contradictions, omitted constraints, bad assumptions, and alternative branches that could beat the obvious path.",
				whyDistinct: "Optimizes for error discovery, semantic drift detection, and robust cross-checking.",
				suggestedChecks: [
					"Find plausible failure modes",
					"Check whether another branch has stronger evidence",
					"Explain what should be killed or ignored",
				],
				risk: "May produce critique without enough constructive progress.",
			},
			{
				title: "Residual expansion path",
				objective:
					"Explore a branch not yet covered by retained residuals while staying deliberately different from the current main plan.",
				whyDistinct:
					"Keeps the swarm from collapsing when fewer live branches remain than available parallel slots.",
				suggestedChecks: [
					"Compare against retained branches",
					"Pursue one uncovered dependency",
					"Return compact evidence even if partial",
				],
				risk: "May be lower-confidence because it is intentionally exploratory.",
			},
		];

		const start = this.branchCounter;
		return Array.from({ length: count }, (_, index) => {
			const template = templates[(start + index) % templates.length];
			const id = `b${requestIndex + 1}.${++this.branchCounter}`;
			return {
				id,
				title: template.title,
				objective: `${template.objective} Current task excerpt: ${userText || "No text excerpt available."}`,
				whyDistinct: template.whyDistinct,
				suggestedChecks: template.suggestedChecks,
				risk: template.risk,
			};
		});
	}

	private buildSubagentSystemPrompt(agentName: string, baseSystemPrompt: string): string {
		return `${baseSystemPrompt}

You are ${agentName}, an autonomous speculative subagent inside a residual swarm.
You are not a fixed-role expert. You receive a branch runtime delta as the latest user message only to encourage diversity; keep agency.
If the branch is wrong, too narrow, or already solved by another residual, switch plans and explicitly state SWITCHED_PLAN with the reason.

Swarm contract:
- Produce a draft, not the final user-facing answer.
- Stay meaningfully different from other branch briefs unless the evidence proves convergence is necessary.
- Use tools when they materially reduce uncertainty and are available to you.
- Do not mutate files or external state unless explicitly allowed by the tool policy.
- Reuse prior branch runtime state, but verify it against the current task before relying on it.
- Identify partial results that may help later rounds even if they are not the best complete answer now.
- End with a compact summary containing: CHOSEN_BRANCH, whether you followed or changed the branch, useful findings, tool/API calls made, unresolved gaps, and collaboration needs.`;
	}

	private buildSubagentRuntimeMessage(roundIndex: number, brief: BranchBrief): string {
		const residualMemory = this.memory
			.slice(-this.options.maxMemoryRounds)
			.map((round) => `Round ${round.roundIndex}: ${round.selection}`)
			.join("\n\n");
		const branchRuntime = this.formatBranchRuntime(brief.id);

		return `<swarm_branch_runtime_delta>
Round: ${roundIndex}
Tool policy: ${this.options.toolPolicy}${this.options.allowBash ? " with bash allowed" : ""}

Assigned branch brief:
- id: ${brief.id}
- title: ${brief.title}
- objective: ${brief.objective}
- why distinct: ${brief.whyDistinct}
- suggested checks: ${brief.suggestedChecks.join("; ")}
- known risk: ${brief.risk}

Prior residual memory:
${residualMemory || "No prior residual memory."}

Branch runtime slot:
${branchRuntime}
</swarm_branch_runtime_delta>`;
	}

	private formatBranchRuntime(briefId: string): string {
		const state = this.branchRuntime.get(briefId);
		if (!state?.lastDraft) {
			return "No completed runtime state for this branch yet.";
		}
		const calls = state.lastDraft.toolCalls
			.slice(0, 8)
			.map((call) => `${call.toolName}(${stableStringify(call.args)})${call.isError ? " ERROR" : ""}`)
			.join("; ");
		return [
			`status=${state.status}`,
			`completed_rounds=${state.completedRounds}`,
			`last_agent=${state.lastAgent ?? "unknown"}`,
			`last_score=${state.lastDraft.score.toFixed(2)}`,
			`last_tools=[${calls || "none"}]`,
			`last_findings=${truncateText(state.lastDraft.finalText.replace(/\s+/g, " ").trim(), 900) || "none"}`,
		].join("\n");
	}

	private buildOrchestratorBriefPrompt(roundIndex: number, briefs: BranchBrief[], baseSystemPrompt: string): string {
		const residualMemory = this.memory
			.slice(-this.options.maxMemoryRounds)
			.map(
				(round) =>
					`Round ${round.roundIndex}: retained=${round.retainedDrafts.map((draft) => draft.briefId).join(",") || "none"}; suppressed=${round.suppressedDrafts.map((draft) => `${draft.briefId}:${draft.eliminationReason}`).join(",") || "none"}`,
			)
			.join("\n");

		const briefText = briefs
			.map(
				(brief, index) =>
					`${index + 1}. ${brief.id} ${brief.title}: ${brief.objective} Distinctness: ${brief.whyDistinct} Risk: ${brief.risk}`,
			)
			.join("\n");

		return `${baseSystemPrompt}

You are the fast Turn0 orchestrator brief agent inside a speculative residual swarm.
You run concurrently with four autonomous subagents. Do not use tools. Do not solve the task fully.
Your job is to provide quick branch prototypes and a voting rubric that later helps the main orchestrator select, merge, or kill branches.
Never provide the final answer, even if the transcript already contains enough evidence. Your output is routing guidance only.

Return a compact brief in this exact shape:
BRANCH_PROTOTYPES:
- branch_id: ...
  vote_for_when: ...
  reject_when: ...
  must_verify: ...
COLLECTIVE_DECISION_RUBRIC:
- prefer branches with real tool/API evidence and exact parameters
- merge branches that repeat the same evidence
- kill branches with entity leakage, semantic drift, or unsupported claims

Round: ${roundIndex}
Candidate branch prototypes:
${briefText}

Prior retained/suppressed branch memory:
${residualMemory || "No prior residual branch memory."}`;
	}

	private buildCollectiveDecision(
		drafts: SwarmDraft[],
		briefs: BranchBrief[],
		preflight: OrchestratorBriefDraft,
		baseMessages: AgentMessage[],
	): CollectiveDecision {
		const briefById = new Map(briefs.map((brief) => [brief.id, brief]));
		const scoredDrafts = drafts.map((draft) => {
			const score = this.scoreDraft(draft, briefById.get(draft.briefId), baseMessages);
			return { ...draft, score };
		});
		const clusters = this.clusterDrafts(scoredDrafts);
		const retainedDrafts: SwarmDraft[] = [];
		const suppressedDrafts: SwarmDraft[] = [];

		for (const [clusterId, cluster] of clusters.entries()) {
			const sorted = cluster
				.map((draft) => ({ ...draft, clusterId }))
				.sort((a, b) => b.score - a.score || b.toolCalls.length - a.toolCalls.length || a.latencyMs - b.latencyMs);
			const winner = sorted[0];
			if (!winner || winner.score < 1) {
				for (const draft of sorted) {
					suppressedDrafts.push({
						...draft,
						eliminationReason:
							draft.eliminationReason ?? "killed: low confidence, empty result, or semantic drift",
					});
				}
				continue;
			}
			retainedDrafts.push(winner);
			for (const duplicate of sorted.slice(1)) {
				suppressedDrafts.push({
					...duplicate,
					eliminationReason: duplicate.eliminationReason ?? `merged: similar to retained branch ${winner.briefId}`,
				});
			}
		}

		const nextBriefs = this.buildNextBriefs(retainedDrafts, suppressedDrafts, briefs, baseMessages);
		const selection = this.formatCollectiveDecision(preflight, briefs, retainedDrafts, suppressedDrafts, nextBriefs);
		return { selection, scoredDrafts, retainedDrafts, suppressedDrafts, nextBriefs };
	}

	private scoreDraft(draft: SwarmDraft, brief: BranchBrief | undefined, baseMessages: AgentMessage[]): number {
		let score = 0;
		if (draft.status === "ok") {
			score += 2;
		}
		if (draft.status === "timeout" || draft.status === "aborted") {
			score -= 1;
		}
		if (draft.status === "error") {
			score -= 2;
		}

		const successfulTools = draft.toolCalls.filter((call) => !call.isError).length;
		const erroredTools = draft.toolCalls.filter((call) => call.isError).length;
		score += Math.min(successfulTools, 4) * 1.25;
		score -= erroredTools * 1.5;

		const normalizedText = normalizeText(draft.finalText);
		if (normalizedText.length > 80) {
			score += 1;
		}
		if (normalizedText.includes("chosen_branch") || normalizedText.includes("switched_plan")) {
			score += 0.5;
		}
		if (brief && this.textOverlap(draft.finalText, `${brief.title} ${brief.objective}`) > 0.08) {
			score += 0.5;
		}
		if (
			this.textOverlap(
				draft.finalText,
				getText(baseMessages[baseMessages.length - 1] ?? ({ role: "user" } as AgentMessage)),
			) < 0.01
		) {
			score -= 0.75;
		}
		if (/(cannot|unable|can't)\s+(access|determine|help|proceed)/i.test(draft.finalText) && successfulTools === 0) {
			score -= 1.5;
		}
		if (hasUnsupportedAbsenceClaim(draft)) {
			score -= 4;
		}
		if (normalizedText.length < 20 && successfulTools === 0) {
			score -= 2;
		}
		return score;
	}

	private clusterDrafts(drafts: SwarmDraft[]): Map<string, SwarmDraft[]> {
		const clusters = new Map<string, SwarmDraft[]>();
		let nextCluster = 1;
		for (const draft of drafts) {
			let matchedCluster: string | undefined;
			for (const [clusterId, cluster] of clusters.entries()) {
				if (cluster.some((existing) => this.areDraftsSimilar(existing, draft))) {
					matchedCluster = clusterId;
					break;
				}
			}
			const clusterId = matchedCluster ?? `c${nextCluster++}`;
			const existing = clusters.get(clusterId) ?? [];
			existing.push(draft);
			clusters.set(clusterId, existing);
		}
		return clusters;
	}

	private areDraftsSimilar(a: SwarmDraft, b: SwarmDraft): boolean {
		const toolSimilarity = jaccard(toolSignatures(a), toolSignatures(b));
		const textSimilarity = jaccard(tokenSet(a.finalText), tokenSet(b.finalText));
		if (a.toolCalls.length > 0 || b.toolCalls.length > 0) {
			return toolSimilarity >= 0.75 || (toolSimilarity >= 0.5 && textSimilarity >= 0.65);
		}
		return textSimilarity >= 0.82;
	}

	private buildNextBriefs(
		retainedDrafts: SwarmDraft[],
		suppressedDrafts: SwarmDraft[],
		previousBriefs: BranchBrief[],
		baseMessages: AgentMessage[],
	): BranchBrief[] {
		const previousById = new Map(previousBriefs.map((brief) => [brief.id, brief]));
		const retainedBriefs = retainedDrafts
			.sort((a, b) => b.score - a.score)
			.map((draft) => previousById.get(draft.briefId))
			.filter((brief): brief is BranchBrief => Boolean(brief));

		const survivingBriefs = dedupeBriefs(retainedBriefs).slice(0, this.options.agents.length);
		const needed = Math.max(0, this.options.agents.length - survivingBriefs.length);
		if (needed === 0) {
			return survivingBriefs;
		}

		const suppressedSummary = suppressedDrafts
			.slice(0, 4)
			.map((draft) => `${draft.briefId}:${draft.eliminationReason ?? "suppressed"}`)
			.join("; ");
		const fresh = this.createFreshBranchBriefs(baseMessages, this.memory.length + 1, needed).map((brief) => ({
			...brief,
			objective: `${brief.objective} Avoid recently suppressed patterns: ${suppressedSummary || "none"}.`,
		}));
		return [...survivingBriefs, ...fresh];
	}

	private formatCollectiveDecision(
		preflight: OrchestratorBriefDraft,
		briefs: BranchBrief[],
		retainedDrafts: SwarmDraft[],
		suppressedDrafts: SwarmDraft[],
		nextBriefs: BranchBrief[],
	): string {
		const toolSupport = new Map<string, { count: number; agents: Set<string>; errors: number }>();
		for (const draft of retainedDrafts) {
			for (const call of draft.toolCalls) {
				const key = `${call.toolName} ${stableStringify(call.args)}`;
				const item = toolSupport.get(key) ?? { count: 0, agents: new Set<string>(), errors: 0 };
				item.count++;
				item.agents.add(draft.agent);
				if (call.isError) {
					item.errors++;
				}
				toolSupport.set(key, item);
			}
		}

		const rankedTools = Array.from(toolSupport.entries())
			.sort((a, b) => b[1].count - a[1].count || a[0].localeCompare(b[0]))
			.slice(0, 12)
			.map(
				([key, value]) =>
					`- ${key} | support=${value.count} agents=${Array.from(value.agents).join(",")} errors=${value.errors}`,
			);

		const retainedSummaries = retainedDrafts.map((draft) => {
			const text = draft.finalText.replace(/\s+/g, " ").trim();
			const preview = truncateText(text, 900);
			return `- keep ${draft.briefId}/${draft.agent}: score=${draft.score.toFixed(2)} cluster=${draft.clusterId ?? "solo"} status=${draft.status} turns=${draft.turns} tools=${draft.toolCalls.length} latency=${draft.latencyMs}ms; ${preview}`;
		});
		const suppressedSummaries = suppressedDrafts.map((draft) => {
			const preview = truncateText(draft.finalText.replace(/\s+/g, " ").trim(), 450);
			return `- suppress ${draft.briefId}/${draft.agent}: score=${draft.score.toFixed(2)} reason=${draft.eliminationReason ?? "merged or low confidence"}; ${preview}`;
		});
		const branchText = briefs
			.map((brief) => `- ${brief.id}: ${brief.title}; objective=${brief.objective}`)
			.join("\n");
		const nextText = nextBriefs.map((brief) => `- ${brief.id}: ${brief.title}; ${brief.whyDistinct}`).join("\n");
		const stage = this.options.twoStage ? "COLLECTIVE SELECTION STAGE" : "SWARM SUMMARY";
		const preflightText = preflight.finalText.includes("BRANCH_PROTOTYPES")
			? truncateText(preflight.finalText.replace(/\s+/g, " ").trim(), 1_200)
			: "Preflight output omitted because it did not follow the branch-prototype rubric. Treat it as non-evidence.";

		return `${stage}
Turn0 orchestrator preflight rubric only: status=${preflight.status} latency=${preflight.latencyMs}ms
${preflightText || "No preflight text."}

Active branch briefs:
${branchText || "- none"}

Retained branches after voting/merge/kill:
${retainedSummaries.length > 0 ? retainedSummaries.join("\n") : "- No branch retained with sufficient confidence."}

Suppressed branches:
${suppressedSummaries.length > 0 ? suppressedSummaries.join("\n") : "- None suppressed."}

Reliable tool/API candidates from retained branches:
${rankedTools.length > 0 ? rankedTools.join("\n") : "- No retained tool/API candidates observed."}

Next-round branch budget:
${nextText || "- none"}`;
	}

	private textOverlap(a: string, b: string): number {
		return jaccard(tokenSet(a), tokenSet(b));
	}

	private buildMemoryMessage(input: {
		requestIndex: number;
		mainModel: Model<any>;
		mainThinkingLevel: ThinkingLevel;
		latencyMs: number;
		latestDrafts: SwarmDraft[];
		latestSelection: string;
	}): string {
		const recentHistory = this.memory
			.slice(-this.options.maxMemoryRounds)
			.map(
				(round) =>
					`Round ${round.roundIndex} request=${round.requestIndex} release=${round.releaseMode} background_complete=${round.backgroundComplete} latency=${round.latencyMs}ms drafts=${round.drafts.length}`,
			)
			.join("\n");

		const rawDrafts = input.latestDrafts
			.map((draft) => {
				const calls = draft.toolCalls
					.slice(0, 10)
					.map((call) => `${call.toolName}(${stableStringify(call.args)})${call.isError ? " ERROR" : ""}`)
					.join("; ");
				return `Agent ${draft.agent}: status=${draft.status}; tool_calls=[${calls || "none"}]; final=${truncateText(draft.finalText, 1_200)}`;
			})
			.join("\n\n");

		return `<residual_swarm_memory version="1" mode="speculative-residual-two-stage">
Main orchestrator model: ${input.mainModel.provider}/${input.mainModel.id}
Main thinking level: ${input.mainThinkingLevel}
Swarm latency: ${input.latencyMs}ms

${input.latestSelection}

RESIDUAL TRANSFER RULES FOR THE MAIN ORCHESTRATOR:
- Treat this block as draft evidence, not ground truth.
- First select reliable context and reusable partial results; then execute the user task using only verified evidence.
- Prefer high-support tool/API calls, but keep single-agent findings when they fill a residual gap.
- Avoid entity leakage: do not mix names, ids, paths, or parameters across unrelated draft branches.
- If a later step needs information already gathered by a draft, reuse that result instead of rediscovering it.
- If drafts conflict, resolve the conflict with the real transcript, available tool results, or a fresh targeted tool call.

Recent residual rounds:
${recentHistory || "No prior residual rounds."}

Compact latest draft evidence:
${rawDrafts}
</residual_swarm_memory>`;
	}

	private traceRound(round: SwarmRound): void {
		const traceFile = process.env.PI_SWARM_TRACE_FILE;
		if (!traceFile) {
			return;
		}
		const payload = {
			type: "swarm_round",
			timestamp: new Date().toISOString(),
			round,
		};
		try {
			appendFileSync(traceFile, `${JSON.stringify(payload)}\n`, "utf8");
		} catch {
			// Tracing is diagnostic only; never fail an agent run because it cannot write.
		}
	}
}

function findLastAssistant(messages: AgentMessage[]): AssistantMessage | undefined {
	for (let i = messages.length - 1; i >= 0; i--) {
		const message = messages[i];
		if (message.role === "assistant") {
			return message as AssistantMessage;
		}
	}
	return undefined;
}

function getText(message: AgentMessage): string {
	const content = "content" in message ? message.content : undefined;
	if (typeof content === "string") {
		return content;
	}
	if (!Array.isArray(content)) {
		return "";
	}
	return content
		.filter((item): item is TextContent => item.type === "text")
		.map((item) => item.text)
		.join("\n");
}

function summarizeToolResult(result: unknown): string | undefined {
	if (!result || typeof result !== "object" || !("content" in result)) {
		return undefined;
	}
	const content = (result as ToolResultMessage).content;
	if (!Array.isArray(content)) {
		return undefined;
	}
	return truncateText(
		content
			.filter((item): item is TextContent => item.type === "text")
			.map((item) => item.text)
			.join("\n"),
		300,
	);
}

function truncateText(text: string, maxChars: number): string {
	if (text.length <= maxChars) {
		return text;
	}
	return `${text.slice(0, Math.max(0, maxChars - 24))}\n...[truncated]`;
}

function normalizeText(text: string): string {
	return text
		.toLowerCase()
		.replace(/[^a-z0-9_\-\s/.:]+/g, " ")
		.replace(/\s+/g, " ")
		.trim();
}

function tokenSet(text: string): Set<string> {
	const tokens = normalizeText(text)
		.split(/\s+/)
		.filter((token) => token.length >= 3);
	return new Set(tokens);
}

function jaccard(a: Set<string>, b: Set<string>): number {
	if (a.size === 0 && b.size === 0) {
		return 1;
	}
	if (a.size === 0 || b.size === 0) {
		return 0;
	}
	let intersection = 0;
	for (const item of a) {
		if (b.has(item)) {
			intersection++;
		}
	}
	return intersection / (a.size + b.size - intersection);
}

function toolSignatures(draft: SwarmDraft): Set<string> {
	return new Set(draft.toolCalls.map((call) => `${call.toolName}:${stableStringify(call.args)}`));
}

function hasUnsupportedAbsenceClaim(draft: SwarmDraft): boolean {
	const text = normalizeText(draft.finalText);
	if (
		!/(not\s+(declared|specified|found|present)|no\s+.*(version|field|entry|result)|missing\s+(version|field|entry|result)|absent)/.test(
			text,
		)
	) {
		return false;
	}
	return draft.toolCalls.some((call) => {
		if (call.toolName !== "read") {
			return false;
		}
		const args = call.args && typeof call.args === "object" ? (call.args as Record<string, unknown>) : {};
		const usedLimit = "limit" in args || "offset" in args;
		const preview = normalizeText(call.resultPreview ?? "");
		return usedLimit || preview.includes("truncated") || preview.includes("more lines");
	});
}

function dedupeBriefs(briefs: BranchBrief[]): BranchBrief[] {
	const seen = new Set<string>();
	const result: BranchBrief[] = [];
	for (const brief of briefs) {
		const key = `${brief.title}:${brief.objective}`;
		if (seen.has(key)) {
			continue;
		}
		seen.add(key);
		result.push(brief);
	}
	return result;
}

function stableStringify(value: unknown): string {
	if (value === null || typeof value !== "object") {
		return JSON.stringify(value);
	}
	if (Array.isArray(value)) {
		return `[${value.map((item) => stableStringify(item)).join(",")}]`;
	}
	const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b));
	return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableStringify(item)}`).join(",")}}`;
}

function trackPromise<T>(promise: Promise<T>): TrackedPromise<T> {
	const tracked: TrackedPromise<T> = {
		settled: false,
		promise: promise
			.then((value) => {
				tracked.settled = true;
				tracked.value = value;
				return value;
			})
			.catch((error) => {
				tracked.settled = true;
				tracked.error = error;
				throw error;
			}),
	};
	return tracked;
}
