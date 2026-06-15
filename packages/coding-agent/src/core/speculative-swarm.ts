import { appendFileSync } from "node:fs";
import {
	Agent,
	type AgentContext,
	type AgentEvent,
	type AgentMessage,
	type AgentTool,
	type AgentToolResult,
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
/** @deprecated All tasks now use the unified dynamic action-DAG scheduler. */
export type SpeculativeSwarmExecutionMode = "auto" | "parallel" | "sequential";

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
	maxDependencyChars?: number;
	executionMode?: SpeculativeSwarmExecutionMode;
	twoStage?: boolean;
	candidateMultiplier?: number;
	waveSize?: number;
	directTakeover?: boolean;
	takeoverThreshold?: number;
	dynamicPolicy?: SpeculativeSwarmDynamicPolicy;
}

interface SwarmToolCallSummary {
	agent: string;
	toolName: string;
	args: unknown;
	isError?: boolean;
	resultPreview?: string;
	resultText?: string;
	stage?: number;
}

interface BranchBrief {
	id: string;
	title: string;
	objective: string;
	whyDistinct: string;
	suggestedChecks: string[];
	risk: string;
	toolName?: string;
	toolDescription?: string;
	parameters?: unknown;
	dependsOn?: string[];
	stableKey?: string;
	priority?: number;
	admission?: number;
	ready?: boolean;
	executionWave?: number;
	recommendedWaveSize?: number;
	laterWaveSize?: number;
	predictedWaveCount?: number;
	planConfidence?: number;
}

export interface SpeculativeSwarmDynamicAction {
	id?: string;
	title: string;
	objective: string;
	whyDistinct?: string;
	suggestedChecks?: string[];
	risk?: string;
	toolName?: string;
	toolDescription?: string;
	parameters?: unknown;
	dependsOn?: string[];
	stableKey?: string;
	priority?: number;
	admission?: number;
	ready?: boolean;
	executionWave?: number;
	recommendedWaveSize?: number;
	laterWaveSize?: number;
	predictedWaveCount?: number;
	planConfidence?: number;
}

export interface SpeculativeSwarmDynamicPlanInput {
	task: string;
	latestObservation: string;
	requestIndex: number;
	maxBranches: number;
	tools: Array<{ name: string; description?: string; parameters?: unknown }>;
	residualMemory: string;
	globalState: {
		taskEpoch: number;
		completedActions: string[];
		toolResults: Array<{ key: string; toolName: string; resultPreview: string }>;
		activeBranches: Array<{ id: string; title: string; status: "active" | "suppressed" }>;
	};
}

export interface SpeculativeSwarmDynamicSelectionInput {
	task: string;
	actions: SpeculativeSwarmDynamicAction[];
	drafts: Array<{
		briefId: string;
		status: SwarmDraft["status"];
		finalText: string;
		toolCalls: SwarmToolCallSummary[];
	}>;
}

export interface SpeculativeSwarmDynamicSelection {
	scores: Array<{
		briefId: string;
		coherence: number;
		utility: number;
		novelty: number;
		retain: boolean;
		failureSource?: number;
		failureType?: string;
	}>;
	nextActions?: SpeculativeSwarmDynamicAction[];
	taskSolvable?: boolean;
	taskComplete?: boolean;
	taskSolvability?: number;
	taskStatus?: "complete" | "incomplete_retryable" | "unsolvable_current_plan";
	mainDecision?: "delegate" | "takeover";
	mainBriefIds?: string[];
	mainConfidence?: number;
	mustReportFailure?: boolean;
	failureReports?: Array<{
		briefId: string;
		toolName?: string;
		failureType: string;
		recoveryAction?: string;
		status?: string;
		blockedBriefIds?: string[];
		message: string;
	}>;
}

export interface SpeculativeSwarmDynamicRouteInput {
	task: string;
	stage: number;
	actions: SpeculativeSwarmDynamicAction[];
	drafts: SpeculativeSwarmDynamicSelectionInput["drafts"];
}

export interface SpeculativeSwarmDynamicRoute {
	routes: Array<{
		targetBriefId: string;
		sourceBriefIds: string[];
		retryOwnErrors?: boolean;
		continueOwn?: boolean;
		ready?: boolean;
		reason?: string;
	}>;
}

export interface SpeculativeSwarmDynamicPolicy {
	plan(input: SpeculativeSwarmDynamicPlanInput): Promise<SpeculativeSwarmDynamicAction[]>;
	select(input: SpeculativeSwarmDynamicSelectionInput): Promise<SpeculativeSwarmDynamicSelection>;
	route?(input: SpeculativeSwarmDynamicRouteInput): Promise<SpeculativeSwarmDynamicRoute>;
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
	stages?: number;
	dependencyUpdates?: number;
	assistantMessage?: AssistantMessage;
}

interface SwarmDependencyTransfer {
	stage: number;
	sourceBriefId: string;
	sourceAgent: string;
	targetBriefId: string;
	targetAgent: string;
	toolName: string;
	args: unknown;
	result: string;
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
	releaseMode: "early" | "barrier" | "background" | "cooperative";
	backgroundComplete: boolean;
	drafts: SwarmDraft[];
	preflight: OrchestratorBriefDraft;
	activeBriefs: BranchBrief[];
	retainedDrafts: SwarmDraft[];
	suppressedDrafts: SwarmDraft[];
	dependencyTransfers: SwarmDependencyTransfer[];
	selection: string;
	mainDecision: "delegate" | "takeover";
	mainBriefIds: string[];
	mainConfidence: number;
	taskStatus?: SpeculativeSwarmDynamicSelection["taskStatus"];
	taskComplete?: boolean;
	taskSolvable?: boolean;
	waves: number;
}

interface CollectiveDecision {
	selection: string;
	scoredDrafts: SwarmDraft[];
	retainedDrafts: SwarmDraft[];
	suppressedDrafts: SwarmDraft[];
	nextBriefs: BranchBrief[];
	mainDecision: "delegate" | "takeover";
	mainBriefIds: string[];
	mainConfidence: number;
	taskStatus?: SpeculativeSwarmDynamicSelection["taskStatus"];
	taskComplete?: boolean;
	taskSolvable?: boolean;
}

interface BranchRuntimeState {
	brief: BranchBrief;
	lastAgent?: string;
	lastDraft?: SwarmDraft;
	completedRounds: number;
	status: "active" | "suppressed";
	updatedAt: number;
}

interface CooperativeBranchExecution {
	agentName: string;
	brief: BranchBrief;
	agent: Agent;
	abortController: AbortController;
	onAbort: () => void;
	startedAt: number;
	turns: number;
	stopAfterTurn: number;
	stage: number;
	timedOut: boolean;
	dependencyUpdates: number;
	toolCalls: Map<string, SwarmToolCallSummary>;
}

interface ToolLedgerEntry {
	key: string;
	toolName: string;
	args: unknown;
	result: AgentToolResult<any>;
	resultPreview: string;
	completedAt: number;
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

export function isLikelySequentialTask(text: string): boolean {
	const normalized = normalizeText(text);
	const strongDependency =
		/\b(?:using|use)\s+(?:the\s+)?(?:result|output|value|id|identifier|symbol|coordinates?|metadata)\s+(?:from|of)\b/.test(
			normalized,
		) ||
		/\b(?:result|output|value|id|identifier|symbol|coordinates?|metadata)\s+from\s+(?:the\s+)?previous\b/.test(
			normalized,
		) ||
		/\b(?:after|once)\s+.+\b(?:then|use|retrieve|find|calculate|check)\b/.test(normalized);
	if (strongDependency) {
		return true;
	}
	const orderingMarkers = [
		/\bfirst\b/,
		/\bthen\b/,
		/\bnext\b/,
		/\bfinally\b/,
		/\bfollowed by\b/,
		/\bafter that\b/,
	].filter((pattern) => pattern.test(normalized)).length;
	return orderingMarkers >= 2;
}

export function extractSequentialSteps(text: string): string[] {
	const cleaned = text.replace(/\s+/g, " ").trim();
	if (!cleaned) {
		return [];
	}

	const parts = cleaned
		.split(/(?:[.;]\s*|,\s*|\s+)(?:and\s+)?(?:then|next|finally|afterwards|subsequently)\b[:,-]?\s*/i)
		.map((part) => part.replace(/^(?:first|initially)\b[:,-]?\s*/i, "").trim())
		.filter((part) => part.length >= 8);
	if (parts.length >= 2) {
		return parts;
	}

	return cleaned
		.split(/[.;]\s+/)
		.map((part) => part.trim())
		.filter((part) => part.length >= 12);
}

export function extractActionCandidates(text: string): string[] {
	const cleaned = text.replace(/\s+/g, " ").trim();
	if (!cleaned) {
		return [];
	}
	const ordered = extractSequentialSteps(cleaned);
	if (ordered.length >= 2) {
		return ordered;
	}
	const clauses = cleaned
		.split(
			/(?:[.;]\s*|,\s+(?=(?:and\s+)?(?:please\s+)?(?:fetch|get|find|look|search|show|list|calculate|check|retrieve|run|call|map|compare)\b))/i,
		)
		.map((part) => part.replace(/^(?:and|also|meanwhile|please)\b[:,-]?\s*/i, "").trim())
		.filter((part) => part.length >= 10);
	return clauses.length >= 2 ? clauses : [cleaned];
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
			| "maxDependencyChars"
			| "executionMode"
			| "twoStage"
			| "candidateMultiplier"
			| "waveSize"
			| "directTakeover"
			| "takeoverThreshold"
		>
	> &
		Pick<SpeculativeSwarmRuntimeOptions, "model" | "thinkingLevel">;
	private memory: SwarmRound[] = [];
	private activeBriefs: BranchBrief[] = [];
	private branchRuntime = new Map<string, BranchRuntimeState>();
	private actionIds = new Map<string, string>();
	private toolLedger = new Map<string, ToolLedgerEntry>();
	private pendingToolExecutions = new Map<string, Promise<AgentToolResult<any>>>();
	private branchCounter = 0;
	private taskEpoch = 0;
	private originalTask = "";
	private latestObservation = "";
	private activePrepare?: Promise<{ context?: AgentContext; response?: AssistantMessage } | undefined>;
	private readonly dynamicPolicy?: SpeculativeSwarmDynamicPolicy;

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
			maxDependencyChars: Math.max(500, options.maxDependencyChars ?? 4_000),
			executionMode: "auto",
			twoStage: options.twoStage ?? true,
			candidateMultiplier: Math.max(1, options.candidateMultiplier ?? 2),
			waveSize: Math.max(1, options.waveSize ?? options.agents?.length ?? DEFAULT_AGENTS.length),
			directTakeover: options.directTakeover ?? true,
			takeoverThreshold: Math.min(1, Math.max(0, options.takeoverThreshold ?? 0.8)),
		};
		this.dynamicPolicy = options.dynamicPolicy;
	}

	async prepareContext(input: SpeculativeSwarmPrepareContext): Promise<AgentContext | undefined> {
		const prepared = await this.prepareTurn(input);
		return prepared?.context;
	}

	async prepareTurn(
		input: SpeculativeSwarmPrepareContext,
	): Promise<{ context?: AgentContext; response?: AssistantMessage } | undefined> {
		if (input.signal?.aborted) {
			return undefined;
		}

		if (this.activePrepare) {
			return this.activePrepare;
		}

		this.activePrepare = this.prepareTurnOnce(input).finally(() => {
			this.activePrepare = undefined;
		});
		return this.activePrepare;
	}

	private async prepareTurnOnce(
		input: SpeculativeSwarmPrepareContext,
	): Promise<{ context?: AgentContext; response?: AssistantMessage } | undefined> {
		const cleanMessages = input.context.messages.filter((message) => !isSpeculativeSwarmMemoryMessage(message));
		const lastMessage = cleanMessages[cleanMessages.length - 1];
		if (!lastMessage || (lastMessage.role !== "user" && lastMessage.role !== "toolResult")) {
			return undefined;
		}
		this.updateGlobalTaskState(cleanMessages, input.requestIndex);
		// Dynamic branches already execute and route their own tool results.
		// Re-running the full swarm for each main-agent tool result duplicates
		// planning, tool calls, and KV work instead of compressing the agent loop.
		if (lastMessage.role === "toolResult" && this.dynamicPolicy) {
			return undefined;
		}

		let latestDrafts: SwarmDraft[] = [];
		let latestSelection = "";
		let latestRound: SwarmRound | undefined;
		const startedAt = Date.now();

		for (let i = 0; i < this.options.rounds; i++) {
			if (input.signal?.aborted) {
				break;
			}

			const round = await this.runSwarmRoundFifo(cleanMessages, input);
			if (!round) {
				break;
			}
			latestRound = round;
			latestDrafts = round.retainedDrafts;
			latestSelection = round.selection;
		}

		if (latestRound) {
			const response = this.buildTakeoverResponse(latestRound, input.model);
			if (response) {
				return { response };
			}
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
			context: {
				...input.context,
				messages: [
					...cleanMessages,
					{
						role: "user",
						content: [{ type: "text", text: memoryMessage }],
						timestamp: Date.now(),
					},
				],
			},
		};
	}

	private async runSwarmRoundFifo(
		cleanMessages: AgentMessage[],
		input: SpeculativeSwarmPrepareContext,
	): Promise<SwarmRound | undefined> {
		return this.runSwarmRoundCooperative(cleanMessages, input);
	}

	private async runSwarmRoundCooperative(
		cleanMessages: AgentMessage[],
		input: SpeculativeSwarmPrepareContext,
	): Promise<SwarmRound | undefined> {
		const roundIndex = this.memory.length + 1;
		const roundStartedAt = Date.now();
		const candidateLimit = this.dynamicPolicy
			? Math.max(1, input.context.tools?.length ?? this.options.agents.length)
			: this.options.agents.length * this.options.candidateMultiplier;
		const candidateBriefs = await this.prepareDynamicActionBriefs(
			cleanMessages,
			input.requestIndex,
			candidateLimit,
			input.context.tools ?? [],
		);
		for (const brief of candidateBriefs) {
			this.ensureBranchRuntime(brief);
		}

		// A trained dynamic policy already produced the action DAG. Running a
		// second LLM planner duplicates work and delays the first executable wave.
		const shouldRunPreflight = !this.dynamicPolicy && this.shouldRunPreflight(input.requestIndex);
		const preflightTask = shouldRunPreflight
			? this.runOrchestratorBrief(roundIndex, cleanMessages, input, candidateBriefs)
			: Promise.resolve(this.createSkippedPreflight(roundIndex, candidateBriefs));
		const replay = this.createRuntimeReplayDraft(roundIndex, candidateBriefs);
		const replayedIds = new Set(replay ? [replay.briefId] : []);
		const preflight = await preflightTask.catch(() =>
			this.createPlaceholderPreflight(roundIndex, candidateBriefs, roundStartedAt),
		);
		let executedBriefs = candidateBriefs.filter((brief) => replayedIds.has(brief.id));
		let drafts = replay ? [replay] : [];
		let dependencyTransfers: SwarmDependencyTransfer[] = [];
		let decision: CollectiveDecision | undefined;
		let waves = 0;
		const executedIds = new Set(replayedIds);
		const completedIds = new Set(
			drafts.filter((draft) => this.isVerifiedCompleteDraft(draft)).map((draft) => draft.briefId),
		);

		while (waves < candidateBriefs.length) {
			const learnedWaveSize =
				waves === 0 ? candidateBriefs[0]?.recommendedWaveSize : candidateBriefs[0]?.laterWaveSize;
			const wave = this.selectExecutionWave(
				candidateBriefs,
				executedIds,
				Math.max(1, learnedWaveSize ?? this.options.waveSize),
				completedIds,
			);
			if (wave.length === 0) {
				break;
			}
			const verifiedEvidence = drafts.filter(
				(draft) => this.isVerifiedCompleteDraft(draft) || this.hasTransferableErrorEvidence(draft),
			);
			const waveMessages =
				waves === 0
					? cleanMessages
					: [
							...cleanMessages,
							{
								role: "user" as const,
								content: [{ type: "text" as const, text: this.buildWaveTransferMessage(verifiedEvidence) }],
								timestamp: Date.now(),
							},
						];
			const result = await this.runCooperativeSubagents(roundIndex, waveMessages, input, wave);
			if (result.drafts.length === 0 && drafts.length === 0) {
				return undefined;
			}
			waves++;
			for (const brief of wave) {
				executedIds.add(brief.id);
			}
			executedBriefs = [...executedBriefs, ...wave];
			drafts = [...drafts, ...result.drafts];
			dependencyTransfers = [...dependencyTransfers, ...result.dependencyTransfers];
			for (const draft of result.drafts) {
				if (
					this.isVerifiedCompleteDraft(draft) ||
					draft.toolCalls.some(
						(call) => Boolean(call.isError) && Boolean((call.resultText ?? call.resultPreview ?? "").trim()),
					)
				) {
					completedIds.add(draft.briefId);
				}
			}
			decision = await this.buildCollectiveDecision(drafts, executedBriefs, preflight, cleanMessages);
			if (candidateBriefs.every((brief) => executedIds.has(brief.id))) {
				break;
			}
			if (input.signal?.aborted) {
				break;
			}
		}
		if (!decision) {
			if (drafts.length === 0) {
				return undefined;
			}
			decision = await this.buildCollectiveDecision(drafts, executedBriefs, preflight, cleanMessages);
		}
		if (candidateBriefs.some((brief) => !executedIds.has(brief.id))) {
			decision = { ...decision, mainDecision: "delegate" };
		}
		const round: SwarmRound = {
			roundIndex,
			requestIndex: input.requestIndex,
			startedAt: roundStartedAt,
			latencyMs: Date.now() - roundStartedAt,
			releaseMode: "cooperative",
			backgroundComplete: true,
			drafts: decision.scoredDrafts,
			preflight,
			activeBriefs: executedBriefs,
			retainedDrafts: decision.retainedDrafts,
			suppressedDrafts: decision.suppressedDrafts,
			dependencyTransfers,
			selection: decision.selection,
			mainDecision: decision.mainDecision,
			mainBriefIds: decision.mainBriefIds,
			mainConfidence: decision.mainConfidence,
			taskStatus: decision.taskStatus,
			taskComplete: decision.taskComplete,
			taskSolvable: decision.taskSolvable,
			waves,
		};
		this.upsertRound(round);
		this.applyDecisionToRuntime(decision, executedBriefs);
		this.traceRound(round);
		return round;
	}

	private async runCooperativeSubagents(
		roundIndex: number,
		baseMessages: AgentMessage[],
		input: SpeculativeSwarmPrepareContext,
		briefs: BranchBrief[],
	): Promise<{ drafts: SwarmDraft[]; dependencyTransfers: SwarmDependencyTransfer[] }> {
		const executions = briefs.map((brief, index) =>
			this.createCooperativeBranchExecution(
				this.options.agents[index % this.options.agents.length],
				roundIndex,
				baseMessages,
				input,
				brief,
			),
		);
		const dependencyTransfers: SwarmDependencyTransfer[] = [];

		try {
			let active = executions.slice();
			for (let stage = 1; stage <= this.options.maxSubagentTurns && active.length > 0; stage++) {
				await Promise.all(active.map((execution) => this.runCooperativeBranchStage(execution)));
				if (stage >= this.options.maxSubagentTurns || input.signal?.aborted) {
					break;
				}

				const stageCalls = executions.flatMap((execution) =>
					Array.from(execution.toolCalls.values())
						.filter((call) => call.stage === stage && Boolean(call.resultText) && !execution.timedOut)
						.map((call) => ({ execution, call })),
				);
				const successfulEvidence = stageCalls.filter(({ call }) => !call.isError);
				let dynamicRoutes: SpeculativeSwarmDynamicRoute["routes"] | undefined;
				if (this.dynamicPolicy?.route) {
					const task = this.originalTask || this.findRootTask(baseMessages);
					const routed = await this.dynamicPolicy.route({
						task,
						stage,
						actions: briefs,
						drafts: executions.map((execution) => {
							const draft = this.finishCooperativeDraft(execution, roundIndex);
							return {
								briefId: draft.briefId,
								status: draft.status,
								finalText: draft.finalText,
								toolCalls: draft.toolCalls,
							};
						}),
					});
					dynamicRoutes = routed.routes;
				} else if (successfulEvidence.length === 0) {
					break;
				}

				const nextActive: CooperativeBranchExecution[] = [];
				for (const target of executions) {
					if (target.timedOut || target.turns >= this.options.maxSubagentTurns) {
						continue;
					}
					if (this.executionHasTerminalProviderError(target)) {
						continue;
					}
					if (this.isExecutionLocallyComplete(target)) {
						continue;
					}
					const route = dynamicRoutes?.find((candidate) => candidate.targetBriefId === target.brief.id);
					const allowedSources = route ? new Set(route.sourceBriefIds) : undefined;
					const peerEvidence = successfulEvidence.filter(
						({ execution }) =>
							execution.brief.id !== target.brief.id &&
							(!allowedSources || allowedSources.has(execution.brief.id)),
					);
					const ownErrors =
						route?.retryOwnErrors && this.canRetryOwnError(target)
							? stageCalls.filter(
									({ execution, call }) => execution.brief.id === target.brief.id && Boolean(call.isError),
								)
							: [];
					const ownSuccess = route?.continueOwn
						? successfulEvidence.filter(({ execution }) => execution.brief.id === target.brief.id)
						: [];
					const routedEvidence = [...peerEvidence, ...ownErrors, ...ownSuccess];
					if (routedEvidence.length === 0) {
						continue;
					}
					const update = this.buildDependencyUpdate(stage, target, routedEvidence, route?.reason);
					target.agent.steer({
						role: "user",
						content: [{ type: "text", text: update }],
						timestamp: Date.now(),
					});
					target.dependencyUpdates++;
					nextActive.push(target);
					for (const { execution: source, call } of peerEvidence) {
						dependencyTransfers.push({
							stage,
							sourceBriefId: source.brief.id,
							sourceAgent: source.agentName,
							targetBriefId: target.brief.id,
							targetAgent: target.agentName,
							toolName: call.toolName,
							args: call.args,
							result: call.resultText ?? "",
						});
					}
				}
				active = nextActive;
			}
		} finally {
			for (const execution of executions) {
				input.signal?.removeEventListener("abort", execution.onAbort);
			}
		}

		return {
			drafts: executions.map((execution) => this.finishCooperativeDraft(execution, roundIndex)),
			dependencyTransfers,
		};
	}

	private createCooperativeBranchExecution(
		agentName: string,
		roundIndex: number,
		baseMessages: AgentMessage[],
		input: SpeculativeSwarmPrepareContext,
		brief: BranchBrief,
	): CooperativeBranchExecution {
		const abortController = new AbortController();
		const execution = {} as CooperativeBranchExecution;
		const agent = new Agent({
			initialState: {
				systemPrompt: this.buildSubagentSystemPrompt(agentName, input.context.systemPrompt),
				model: this.options.model,
				thinkingLevel: this.options.thinkingLevel ?? "off",
				tools: this.filterToolsForBrief(input.context.tools ?? [], brief),
				messages: [
					...this.sanitizeBranchMessages(baseMessages, brief),
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
			shouldStopAfterTurn: () => execution.turns >= execution.stopAfterTurn,
			sessionId: `swarm-branch-${brief.id}-${agentName}`,
			transport: "auto",
		});
		Object.assign(execution, {
			agentName,
			brief,
			agent,
			abortController,
			startedAt: Date.now(),
			turns: 0,
			stopAfterTurn: 1,
			stage: 1,
			timedOut: false,
			dependencyUpdates: 0,
			toolCalls: new Map<string, SwarmToolCallSummary>(),
		});
		const onAbort = () => {
			abortController.abort();
			agent.abort();
		};
		execution.onAbort = onAbort;
		input.signal?.addEventListener("abort", onAbort, { once: true });
		agent.subscribe((event) => {
			this.captureSubagentEvent(agentName, event, execution.toolCalls, execution.stage);
			if (event.type === "turn_end") {
				execution.turns++;
			}
		});
		return execution;
	}

	private async runCooperativeBranchStage(execution: CooperativeBranchExecution): Promise<void> {
		execution.stopAfterTurn = execution.turns + 1;
		execution.stage = execution.turns + 1;
		const promise = execution.agent.continue();
		execution.timedOut =
			execution.timedOut ||
			(await this.runWithTimeout(promise, execution.abortController, this.options.timeoutMs, () =>
				execution.agent.abort(),
			));
	}

	private buildDependencyUpdate(
		stage: number,
		target: CooperativeBranchExecution,
		peerEvidence: Array<{ execution: CooperativeBranchExecution; call: SwarmToolCallSummary }>,
		routeReason?: string,
	): string {
		const evidence = peerEvidence
			.map(
				({ execution, call }) =>
					`- source_branch=${execution.brief.id} source_agent=${execution.agentName}\n` +
					`  outcome=${call.isError ? "error" : "success"}\n` +
					`  tool=${call.toolName}\n` +
					`  args=${stableStringify(call.args)}\n` +
					`  result=${truncateText(call.resultText ?? "", this.options.maxDependencyChars)}`,
			)
			.join("\n");
		return `<swarm_dependency_update stage="${stage}" target_branch="${target.brief.id}">
RCG routed the following execution evidence to your assigned branch.
Route reason: ${routeReason ?? "Dependency evidence or local repair signal."}

${evidence}

Continuation rules:
- Stay on assigned branch ${target.brief.id}: ${target.brief.title}.
- Use a sibling result only when it supplies a missing parameter, entity, constraint, or prerequisite.
- Do not repeat a sibling tool call merely to rediscover the same result.
- Treat outcome=error as a repair signal, never as factual task evidence.
- Repair any speculative or failed call from the previous stage with the new evidence.
- If your branch is already complete, return DEPENDENCY_NOOP and a compact reusable finding.
</swarm_dependency_update>`;
	}

	private finishCooperativeDraft(execution: CooperativeBranchExecution, roundIndex: number): SwarmDraft {
		const finalAssistant = findLastAssistant(execution.agent.state.messages);
		const assistantText = finalAssistant ? getText(finalAssistant) : "";
		const toolEvidence = Array.from(execution.toolCalls.values())
			.filter((call) => Boolean(call.resultText))
			.map(
				(call) =>
					`${call.toolName}(${stableStringify(call.args)}) => ` +
					`${call.isError ? "ERROR: " : ""}${call.resultText}`,
			)
			.join("\n");
		const finalText = assistantText.trim() || toolEvidence;
		const status = execution.timedOut
			? "timeout"
			: execution.abortController.signal.aborted
				? "aborted"
				: finalAssistant?.stopReason === "error"
					? "error"
					: "ok";
		const draft: SwarmDraft = {
			agent: execution.agentName,
			roundIndex,
			briefId: execution.brief.id,
			branchTitle: execution.brief.title,
			latencyMs: Date.now() - execution.startedAt,
			turns: execution.turns,
			status,
			finalText: truncateText(finalText, this.options.maxDraftChars),
			toolCalls: Array.from(execution.toolCalls.values()),
			score: 0,
			usage: finalAssistant?.usage,
			errorMessage: finalAssistant?.errorMessage,
			stages: execution.stage,
			dependencyUpdates: execution.dependencyUpdates,
		};
		return attachAssistantMessage(draft, finalAssistant);
	}

	private isExecutionLocallyComplete(execution: CooperativeBranchExecution): boolean {
		const relevant = Array.from(execution.toolCalls.values()).filter(
			(call) => !execution.brief.toolName || call.toolName === execution.brief.toolName,
		);
		if (relevant.length > 0) {
			return relevant.every((call) => call.isError === false && Boolean(call.resultText));
		}
		const finalAssistant = findLastAssistant(execution.agent.state.messages);
		const text = finalAssistant ? getText(finalAssistant).trim() : "";
		return Boolean(
			text &&
				!/\b(?:blocked|dependency_noop|waiting for|missing prerequisite)\b/i.test(text) &&
				finalAssistant?.stopReason === "stop",
		);
	}

	private canRetryOwnError(execution: CooperativeBranchExecution): boolean {
		const errors = Array.from(execution.toolCalls.values()).filter((call) => call.isError);
		if (errors.length === 0) {
			return false;
		}
		const repairablePattern =
			/\b(?:argument|parameter|schema|json|parse|format|type|validation|missing required|invalid call)\b/i;
		if (this.executionHasTerminalProviderError(execution)) {
			return false;
		}
		const errorStages = new Set(errors.map((call) => call.stage));
		return (
			errorStages.size < 2 &&
			errors.every((call) =>
				repairablePattern.test(`${call.resultText ?? ""}\n${call.resultPreview ?? ""}`),
			)
		);
	}

	private executionHasTerminalProviderError(execution: CooperativeBranchExecution): boolean {
		const terminalPattern =
			/\b(?:401|403|404|408|409|429|500|502|503|504|auth|quota|rate.?limit|timeout|timed out|unavailable|not found|connection|network|service)\b/i;
		return Array.from(execution.toolCalls.values()).some(
			(call) =>
				call.isError &&
				terminalPattern.test(`${call.resultText ?? ""}\n${call.resultPreview ?? ""}`),
		);
	}

	private hasTransferableErrorEvidence(draft: SwarmDraft): boolean {
		return draft.toolCalls.some(
			(call) =>
				call.isError &&
				Boolean((call.resultText ?? call.resultPreview ?? "").trim()),
		);
	}

	private isVerifiedCompleteDraft(draft: SwarmDraft): boolean {
		if (draft.status !== "ok") {
			return false;
		}
		if (draft.toolCalls.length > 0) {
			return draft.toolCalls.every((call) => call.isError === false && Boolean(call.resultText));
		}
		return Boolean(
			draft.finalText.trim() &&
				!/\b(?:blocked|dependency_noop|waiting for|missing prerequisite)\b/i.test(draft.finalText),
		);
	}

	private createRuntimeReplayDraft(roundIndex: number, activeBriefs: BranchBrief[]): SwarmDraft | undefined {
		if (process.env.PI_SWARM_ENABLE_RUNTIME_REPLAY === "0") {
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
			if (draft.score < 2.2 || draft.status !== "ok" || hasUnsupportedAbsenceClaim(draft)) {
				continue;
			}
			if (normalizeText(draft.finalText).length < 160 && draft.toolCalls.length === 0) {
				continue;
			}
			return attachAssistantMessage(
				{
					...draft,
					agent: `RuntimeReplay:${state.lastAgent ?? draft.agent}`,
					roundIndex,
					latencyMs: 0,
					turns: 0,
					score: Math.max(1, draft.score - 0.25),
				},
				draft.assistantMessage,
			);
		}
		return undefined;
	}

	private shouldRunPreflight(requestIndex: number): boolean {
		return requestIndex === 0 && this.memory.length === 0;
	}

	private updateGlobalTaskState(messages: AgentMessage[], requestIndex: number): void {
		if (requestIndex === 0) {
			this.taskEpoch++;
			this.originalTask = this.findRootTask(messages);
			this.latestObservation = this.originalTask;
			this.memory = [];
			this.activeBriefs = [];
			this.branchRuntime.clear();
			this.actionIds.clear();
			this.toolLedger.clear();
			this.pendingToolExecutions.clear();
			return;
		}
		const latest = messages[messages.length - 1];
		this.latestObservation = latest ? getText(latest) : "";
	}

	private findRootTask(messages: AgentMessage[]): string {
		for (let index = messages.length - 1; index >= 0; index--) {
			const message = messages[index];
			if (message.role === "user" && !isSpeculativeSwarmMemoryMessage(message)) {
				return getText(message);
			}
		}
		return "";
	}

	private stableActionId(stableKey: string, _preferredId?: string): string {
		const normalized = normalizeText(stableKey) || `action-${++this.branchCounter}`;
		const existing = this.actionIds.get(normalized);
		if (existing) {
			return existing;
		}
		const toolSlug =
			normalized
				.split(/[:\s]/, 1)[0]
				.replace(/[^a-z0-9_-]/g, "")
				.slice(0, 24) || "action";
		const id = `rcg-${toolSlug}-${shortHash(normalized)}`;
		this.actionIds.set(normalized, id);
		return id;
	}

	private selectExecutionWave(
		candidates: BranchBrief[],
		executedIds: Set<string>,
		limit: number,
		completedIds = new Set<string>(),
	): BranchBrief[] {
		const ready = candidates
			.filter((brief) => !executedIds.has(brief.id))
			.filter((brief) =>
				brief.dependsOn?.length
					? brief.dependsOn.every((id) => completedIds.has(id) || !candidates.some((item) => item.id === id))
					: brief.ready !== false,
			);
		const earliestLearnedWave = ready.length
			? Math.min(...ready.map((brief) => Math.max(1, brief.executionWave ?? 1)))
			: 1;
		return ready
			.filter((brief) => Math.max(1, brief.executionWave ?? 1) <= earliestLearnedWave)
			.sort((left, right) => (right.priority ?? right.admission ?? 0) - (left.priority ?? left.admission ?? 0))
			.slice(0, limit);
	}

	private buildWaveTransferMessage(retainedDrafts: SwarmDraft[]): string {
		const evidence = retainedDrafts
			.map(
				(draft) =>
					`- branch=${draft.briefId} status=${draft.status}\n` +
					`  result=${truncateText(draft.finalText, this.options.maxDependencyChars)}\n` +
					`  tools=${draft.toolCalls
						.filter((call) => !call.isError)
						.map(
							(call) =>
								`${call.toolName}(${stableStringify(call.args)})=${truncateText(call.resultText ?? "", 600)}`,
						)
						.join("; ")}`,
			)
			.join("\n");
		return `<swarm_wave_transfer task_epoch="${this.taskEpoch}">
Verified evidence retained from the previous speculative wave:
${evidence || "- none"}

Use this evidence as shared runtime state. Do not repeat an identical successful tool call.
</swarm_wave_transfer>`;
	}

	private formatVerifiedDraft(draft: SwarmDraft): string {
		if (draft.toolCalls.length === 0) {
			return draft.finalText
				.split("\n")
				.filter((line) => !/^\s*UNTRUSTED_DIAGNOSTIC_NOISE\b/i.test(line))
				.join("\n")
				.trim();
		}
		return draft.toolCalls
			.map((call) => {
				const result = (call.resultText ?? call.resultPreview ?? "")
					.split("\n")
					.filter((line) => !/^\s*UNTRUSTED_DIAGNOSTIC_NOISE\b/i.test(line))
					.join("\n")
					.trim();
				const minuteFacts = Array.from(
					result.matchAll(/\b[A-Z][A-Z0-9_]*_MINUTES=(-?\d+(?:\.\d+)?)\b/g),
					(match) => `${match[1]} minutes`,
				);
				const annotation = minuteFacts.length > 0 ? ` (${minuteFacts.join(", ")})` : "";
				return `${call.toolName}(${stableStringify(call.args)}) => ${result}${annotation}`;
			})
			.join("\n");
	}

	private buildTakeoverResponse(round: SwarmRound, _mainModel: Model<any>): AssistantMessage | undefined {
		if (
			round.mainDecision !== "takeover" ||
			round.taskComplete !== true ||
			round.mainConfidence < this.options.takeoverThreshold ||
			round.mainBriefIds.length === 0
		) {
			return undefined;
		}
		const selected = round.mainBriefIds
			.map((id) => round.retainedDrafts.find((draft) => draft.briefId === id))
			.filter((draft): draft is SwarmDraft => Boolean(draft));
		const source = selected.find((draft) => draft.assistantMessage)?.assistantMessage;
		if (!source || selected.length === 0) {
			return undefined;
		}
		const text =
			round.taskSolvable === false
				? `${round.selection.split("\n\nCOLLECTIVE SELECTION STAGE", 1)[0].trim()}

VERIFIED_PARTIAL_RESULTS:
${selected.map((draft) => `- ${draft.branchTitle}: ${this.formatVerifiedDraft(draft)}`).join("\n")}`
				: selected.length === 1
					? this.formatVerifiedDraft(selected[0])
					: selected
							.map((draft) => `[${draft.branchTitle}]\n${this.formatVerifiedDraft(draft)}`)
							.join("\n\n");
		if (!text.trim()) {
			return undefined;
		}
		return {
			...source,
			content: [{ type: "text", text }],
			stopReason: "stop",
			errorMessage: undefined,
			timestamp: Date.now(),
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
		const existing = this.branchRuntime.get(brief.id);
		if (existing) {
			existing.brief = brief;
			existing.updatedAt = Date.now();
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
		let agent: Agent;
		const onAbort = () => {
			abortController.abort();
			agent?.abort();
		};
		input.signal?.addEventListener("abort", onAbort, { once: true });

		agent = new Agent({
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
			() => agent.abort(),
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
		onTimeout?: () => void,
	): Promise<boolean> {
		let timeout: NodeJS.Timeout | undefined;
		const timeoutPromise = new Promise<"timeout">((resolve) => {
			timeout = setTimeout(() => {
				abortController.abort();
				onTimeout?.();
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
		stage: number,
	): void {
		if (event.type === "tool_execution_start") {
			toolCalls.set(event.toolCallId, {
				agent: agentName,
				toolName: event.toolName,
				args: event.args,
				stage,
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
				resultText: summarizeToolResult(event.result, this.options.maxDependencyChars),
				stage: existing?.stage ?? stage,
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

	private sanitizeBranchMessages(messages: AgentMessage[], brief: BranchBrief): AgentMessage[] {
		return messages.flatMap<AgentMessage>((message): AgentMessage[] => {
			if (isSpeculativeSwarmMemoryMessage(message)) {
				return [];
			}
			if (message.role === "toolResult") {
				return !brief.toolName || message.toolName === brief.toolName ? [message] : [];
			}
			if (message.role !== "assistant") {
				return [message];
			}
			const content = message.content.filter((item) => item.type !== "toolCall");
			return content.length > 0 ? [{ ...message, content }] : [];
		});
	}

	private filterToolsForBrief(tools: AgentTool<any>[], brief: BranchBrief): AgentTool<any>[] {
		const available = this.filterTools(tools);
		if (!brief.toolName) {
			return available.map((tool) => this.wrapToolWithLedger(tool));
		}
		const assigned = available.filter((tool) => tool.name === brief.toolName);
		return assigned.map((tool) => this.wrapToolWithLedger(tool));
	}

	private wrapToolWithLedger(tool: AgentTool<any>): AgentTool<any> {
		return {
			...tool,
			execute: async (toolCallId, params, signal, onUpdate) => {
				const key = `${tool.name}:${stableStringify(params)}`;
				const cached = this.toolLedger.get(key);
				if (cached) {
					return cached.result;
				}
				const pending = this.pendingToolExecutions.get(key);
				if (pending) {
					return pending;
				}
				const execution = tool.execute(toolCallId, params, signal, onUpdate);
				this.pendingToolExecutions.set(key, execution);
				try {
					const result = await execution;
					this.toolLedger.set(key, {
						key,
						toolName: tool.name,
						args: params,
						result,
						resultPreview: summarizeAgentToolResult(result, this.options.maxDependencyChars),
						completedAt: Date.now(),
					});
					return result;
				} finally {
					this.pendingToolExecutions.delete(key);
				}
			},
		};
	}

	private prepareBranchBriefs(baseMessages: AgentMessage[], requestIndex: number, count: number): BranchBrief[] {
		const retained = this.activeBriefs.slice(0, count);
		const needed = Math.max(0, count - retained.length);
		if (needed === 0) {
			return retained;
		}
		return [...retained, ...this.createFreshBranchBriefs(baseMessages, requestIndex, needed)];
	}

	private async prepareDynamicActionBriefs(
		baseMessages: AgentMessage[],
		requestIndex: number,
		count: number,
		tools: AgentTool[],
	): Promise<BranchBrief[]> {
		const taskText = this.originalTask || this.findRootTask(baseMessages);
		if (this.dynamicPolicy) {
			const planned = await this.dynamicPolicy.plan({
				task: taskText,
				latestObservation: this.latestObservation,
				requestIndex,
				maxBranches: count,
				tools: tools.map((tool) => ({
					name: tool.name,
					description: tool.description,
					parameters: tool.parameters,
				})),
				residualMemory: this.memory
					.slice(-this.options.maxMemoryRounds)
					.map((round) => round.selection)
					.join("\n"),
				globalState: {
					taskEpoch: this.taskEpoch,
					completedActions: Array.from(this.branchRuntime.values())
						.filter((state) => state.status === "active" && state.lastDraft?.status === "ok")
						.map((state) => state.brief.id),
					toolResults: Array.from(this.toolLedger.values()).map((entry) => ({
						key: entry.key,
						toolName: entry.toolName,
						resultPreview: entry.resultPreview,
					})),
					activeBranches: Array.from(this.branchRuntime.values()).map((state) => ({
						id: state.brief.id,
						title: state.brief.title,
						status: state.status,
					})),
				},
			});
			if (planned.length > 0) {
				const idAliases = new Map<string, string>();
				const prepared = planned.slice(0, count).map((action, index) => {
					const stableKey =
						action.stableKey ??
						`${action.toolName ?? action.title}:${normalizeText(action.objective || action.title)}`;
					const id = this.stableActionId(stableKey, action.id);
					if (action.id) {
						idAliases.set(action.id, id);
					}
					return {
						id,
						title: action.title || `Action node ${index + 1}`,
						objective: action.objective,
						whyDistinct: action.whyDistinct ?? "Selected by the dynamic RCG action policy.",
						suggestedChecks: action.suggestedChecks ?? [],
						risk: action.risk ?? "Controller confidence should be re-evaluated after tool execution.",
						toolName: action.toolName,
						toolDescription: action.toolDescription,
						parameters: action.parameters,
						dependsOn: action.dependsOn,
						stableKey,
						priority: action.priority,
						admission: action.admission,
						ready: action.ready,
						executionWave: action.executionWave,
						recommendedWaveSize: action.recommendedWaveSize,
						laterWaveSize: action.laterWaveSize,
						predictedWaveCount: action.predictedWaveCount,
						planConfidence: action.planConfidence,
					};
				});
				return prepared.map((brief) => ({
					...brief,
					dependsOn: brief.dependsOn?.map((id) => idAliases.get(id) ?? id),
				}));
			}
		}
		const extracted = extractActionCandidates(taskText);
		if (extracted.length < 2) {
			return this.prepareBranchBriefs(baseMessages, requestIndex, count);
		}

		const steps =
			extracted.length <= count
				? extracted
				: [...extracted.slice(0, count - 1), extracted.slice(count - 1).join(" Then continue with: ")];
		const briefs = steps.map((step, index): BranchBrief => {
			const suggestedTools = this.matchToolsForStep(step, tools);
			const toolHint =
				suggestedTools.length > 0
					? ` Candidate tools: ${suggestedTools.join(", ")}.`
					: " Inspect the available tool set for the narrowest matching action.";
			return {
				id: `b${requestIndex + 1}.${++this.branchCounter}`,
				title: `Action node ${index + 1}`,
				objective:
					`Own and pre-execute only this action: ${truncateText(step, 500)}.${toolHint} ` +
					"Use explicit query values immediately. Infer whether inputs are ready from the current state; if a prerequisite is unavailable, preserve partial work, name the missing field, and wait for a routed sibling update.",
				whyDistinct: `Owns one semantic action node instead of solving the full task or duplicating another node.`,
				suggestedChecks: [
					"Identify inputs already explicit in the query",
					"Separate missing upstream inputs from locally available inputs",
					"Return exact tool output or a precise blocked-input declaration",
				],
				risk: "A speculative call may be invalid until an earlier branch returns its real output.",
			};
		});

		return briefs;
	}

	private matchToolsForStep(step: string, tools: AgentTool[]): string[] {
		const stepTerms = tokenSet(step);
		return tools
			.map((tool) => {
				const toolTerms = tokenSet(`${tool.name} ${tool.description}`);
				let score = 0;
				for (const term of stepTerms) {
					if (toolTerms.has(term)) {
						score++;
					}
				}
				return { name: tool.name, score };
			})
			.filter((item) => item.score > 0)
			.sort((a, b) => b.score - a.score || a.name.localeCompare(b.name))
			.slice(0, 2)
			.map((item) => item.name);
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
- Treat timeout, authentication, quota, server, unavailable-tool, and malformed-result failures as failed local objectives.
- Never replace a failed tool result with a plausible guess. Name the failed tool and failure class so the main agent can report an incomplete task.
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
Scheduling mode: unified dynamic action-DAG auto

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

Dynamic action-DAG rules:
- Treat the assigned branch as your owned action slot, not as a request to solve the whole task.
- You may pre-execute with explicit query values in stage 1.
- When an upstream value is missing, preserve your partial work and name the exact missing parameter.
- A later <swarm_dependency_update> continues this same agent session; use it to repair or complete the owned action without restarting from scratch.
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

	private async buildCollectiveDecision(
		drafts: SwarmDraft[],
		briefs: BranchBrief[],
		preflight: OrchestratorBriefDraft,
		baseMessages: AgentMessage[],
	): Promise<CollectiveDecision> {
		if (this.dynamicPolicy) {
			const task = this.originalTask || this.findRootTask(baseMessages);
			const selected = await this.dynamicPolicy.select({
				task,
				actions: briefs,
				drafts: drafts.map((draft) => ({
					briefId: draft.briefId,
					status: draft.status,
					finalText: draft.finalText,
					toolCalls: draft.toolCalls,
				})),
			});
			const byId = new Map(selected.scores.map((score) => [score.briefId, score]));
			const scoredDrafts = drafts.map((draft) => {
				const score = byId.get(draft.briefId);
				return attachAssistantMessage(
					{
						...draft,
						score: score ? score.coherence + score.utility + score.novelty : 0,
					},
					draft.assistantMessage,
				);
			});
			const retainedDrafts = scoredDrafts.filter((draft) => byId.get(draft.briefId)?.retain);
			const suppressedDrafts = scoredDrafts
				.filter((draft) => !byId.get(draft.briefId)?.retain)
				.map((draft) => ({
					...draft,
					eliminationReason: "RCG coherence/utility/novelty policy rejected this path.",
				}));
			const policyNextActions: SpeculativeSwarmDynamicAction[] =
				selected.nextActions ??
				retainedDrafts.map((draft) => {
					const brief = briefs.find((candidate) => candidate.id === draft.briefId);
					return brief
						? { ...brief }
						: {
								title: draft.branchTitle,
								objective: draft.finalText,
							};
				});
			const nextBriefs = policyNextActions.map((action, index) => ({
				id: this.stableActionId(
					action.stableKey ??
						`${action.toolName ?? action.title}:${normalizeText(action.objective || action.title)}`,
					action.id,
				),
				title: action.title || `Action node ${index + 1}`,
				objective: action.objective,
				whyDistinct: action.whyDistinct ?? "Retained by the dynamic RCG policy.",
				suggestedChecks: action.suggestedChecks ?? [],
				risk: action.risk ?? "Re-score after the next tool result.",
				toolName: action.toolName,
				toolDescription: action.toolDescription,
				parameters: action.parameters,
				dependsOn: action.dependsOn,
				stableKey: action.stableKey,
				priority: action.priority,
				admission: action.admission,
				ready: action.ready,
				executionWave: action.executionWave,
				recommendedWaveSize: action.recommendedWaveSize,
				laterWaveSize: action.laterWaveSize,
				predictedWaveCount: action.predictedWaveCount,
				planConfidence: action.planConfidence,
			}));
			let selection = this.formatCollectiveDecision(preflight, briefs, retainedDrafts, suppressedDrafts, nextBriefs);
			if (selected.taskSolvable === false) {
				const failures = (selected.failureReports ?? [])
					.map(
						(failure) =>
							`- ${failure.briefId} | ${failure.toolName ?? "unknown tool"} | ${failure.failureType}` +
							`${failure.recoveryAction ? ` | recovery=${failure.recoveryAction}` : ""}` +
							`${failure.blockedBriefIds?.length ? ` | blocks=${failure.blockedBriefIds.join(",")}` : ""}`,
					)
					.join("\n");
				const completedWithFailure = selected.taskComplete === true || selected.taskStatus === "complete";
				selection = `TASK_STATUS: ${
					completedWithFailure
						? "COMPLETE_WITH_REPORTED_FAILURE"
						: selected.taskStatus === "incomplete_retryable"
							? "INCOMPLETE_RETRYABLE"
							: "UNSOLVABLE_CURRENT_PLAN"
				}
${
	completedWithFailure
		? "The swarm completed the truthful response using verified successes and attributed failure evidence."
		: "The swarm did not obtain every required result. The main agent must not fabricate a complete answer before recovery succeeds."
}
FAILED_STEPS:
${failures || "- RCG marked the task unsolvable but did not return a branch attribution."}
Required response: preserve verified partial evidence, identify failed components precisely, and never invent a successful tool result.

${selection}`;
			}
			const retainedIds = new Set(retainedDrafts.map((draft) => draft.briefId));
			const structurallySuccessfulBriefIds = new Set(
				retainedDrafts
					.filter((draft) => {
						const brief = briefs.find((candidate) => candidate.id === draft.briefId);
						if (!brief?.toolName) {
							return this.isVerifiedCompleteDraft(draft);
						}
						const expectedCalls = draft.toolCalls.filter(
							(call) => call.toolName === brief.toolName,
						);
						return (
							expectedCalls.length > 0 &&
							expectedCalls.every(
								(call) =>
									!call.isError &&
									Boolean((call.resultText ?? call.resultPreview ?? "").trim()),
							)
						);
					})
					.map((draft) => draft.briefId),
			);
			const allPlannedActionsSucceeded =
				briefs.length > 0 &&
				briefs.every((brief) => structurallySuccessfulBriefIds.has(brief.id));
			const structurallyFailedBriefIds = new Set(
				retainedDrafts
					.filter((draft) => {
						const brief = briefs.find((candidate) => candidate.id === draft.briefId);
						return Boolean(
							brief?.toolName &&
								draft.toolCalls.some(
									(call) =>
										call.toolName === brief.toolName &&
										call.isError &&
										Boolean((call.resultText ?? call.resultPreview ?? "").trim()),
								),
						);
					})
					.map((draft) => draft.briefId),
			);
			const recoveredFailureIds = new Set(
				briefs
					.filter(
						(brief) =>
							structurallySuccessfulBriefIds.has(brief.id) &&
							(brief.dependsOn ?? []).some((id) => structurallyFailedBriefIds.has(id)),
					)
					.flatMap((brief) => brief.dependsOn ?? []),
			);
			const allPlannedActionsAccountedFor =
				briefs.length > 0 &&
				briefs.every(
					(brief) =>
						structurallySuccessfulBriefIds.has(brief.id) ||
						structurallyFailedBriefIds.has(brief.id),
				);
			const structurallyRecovered =
				allPlannedActionsAccountedFor &&
				structurallyFailedBriefIds.size > 0 &&
				Array.from(structurallyFailedBriefIds).every((id) => recoveredFailureIds.has(id));
			const structuralTakeover = allPlannedActionsSucceeded || structurallyRecovered;
			const requestedMainIds = (selected.mainBriefIds ?? []).filter((id) => retainedIds.has(id));
			const fallbackMain = retainedDrafts
				.slice()
				.sort((left, right) => right.score - left.score || left.latencyMs - right.latencyMs)[0];
			const mainBriefIds = structuralTakeover
				? retainedDrafts.map((draft) => draft.briefId)
				: requestedMainIds.length
					? requestedMainIds
					: fallbackMain
						? [fallbackMain.briefId]
						: [];
			const fallbackConfidence = fallbackMain ? Math.min(1, Math.max(0, fallbackMain.score / 3)) : 0;
			const mainConfidence = structuralTakeover
				? 1
				: (selected.mainConfidence ?? fallbackConfidence);
			const policyComplete =
				selected.taskComplete === true ||
				selected.taskStatus === "complete" ||
				structuralTakeover;
			const structurallyComplete =
				policyComplete &&
				mainBriefIds.length > 0 &&
				mainBriefIds.every((id) => {
					const draft = retainedDrafts.find((candidate) => candidate.briefId === id);
					return Boolean(
						draft &&
							(draft.finalText.trim() ||
								draft.toolCalls.some((call) => Boolean((call.resultText ?? call.resultPreview ?? "").trim()))),
					);
				});
			const mainDecision =
				this.options.directTakeover &&
				(selected.mainDecision === "takeover" || structurallyComplete) &&
				structurallyComplete &&
				mainConfidence >= this.options.takeoverThreshold
					? "takeover"
					: "delegate";
			return {
				selection,
				scoredDrafts,
				retainedDrafts,
				suppressedDrafts,
				nextBriefs,
				mainDecision,
				mainBriefIds,
				mainConfidence,
				taskStatus: structuralTakeover ? "complete" : selected.taskStatus,
				taskComplete: policyComplete,
				taskSolvable: structurallyRecovered ? true : selected.taskSolvable,
			};
		}
		const briefById = new Map(briefs.map((brief) => [brief.id, brief]));
		const scoredDrafts = drafts.map((draft) => {
			const score = this.scoreDraft(draft, briefById.get(draft.briefId), baseMessages);
			return attachAssistantMessage({ ...draft, score }, draft.assistantMessage);
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
		return {
			selection,
			scoredDrafts,
			retainedDrafts,
			suppressedDrafts,
			nextBriefs,
			mainDecision: "delegate",
			mainBriefIds: [],
			mainConfidence: 0,
		};
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
					`Round ${round.roundIndex} request=${round.requestIndex} release=${round.releaseMode} background_complete=${round.backgroundComplete} latency=${round.latencyMs}ms drafts=${round.drafts.length} dependency_transfers=${round.dependencyTransfers.length}`,
			)
			.join("\n");

		const rawDrafts = input.latestDrafts
			.map((draft) => {
				const calls = draft.toolCalls
					.slice(0, 10)
					.map((call) => `${call.toolName}(${stableStringify(call.args)})${call.isError ? " ERROR" : ""}`)
					.join("; ");
				return `Agent ${draft.agent}: status=${draft.status}; stages=${draft.stages ?? 1}; dependency_updates=${draft.dependencyUpdates ?? 0}; tool_calls=[${calls || "none"}]; final=${truncateText(draft.finalText, 1_200)}`;
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
- If TASK_STATUS is UNSOLVABLE, do not synthesize a complete answer. Report each failed required step and any downstream step it blocked.
- Timeout, authentication, quota, network, server, missing-tool, and invalid-response failures are evidence of non-completion, not permission to infer the missing result.
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

function summarizeToolResult(result: unknown, maxChars = 300): string | undefined {
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
		maxChars,
	);
}

function summarizeAgentToolResult(result: AgentToolResult<any>, maxChars = 300): string {
	return truncateText(
		result.content
			.filter((item): item is TextContent => item.type === "text")
			.map((item) => item.text)
			.join("\n"),
		maxChars,
	);
}

function attachAssistantMessage(draft: SwarmDraft, message: AssistantMessage | undefined): SwarmDraft {
	if (message) {
		Object.defineProperty(draft, "assistantMessage", {
			value: message,
			enumerable: false,
			configurable: false,
			writable: false,
		});
	}
	return draft;
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

function shortHash(value: string): string {
	let hash = 0x811c9dc5;
	for (let index = 0; index < value.length; index++) {
		hash ^= value.charCodeAt(index);
		hash = Math.imul(hash, 0x01000193);
	}
	return (hash >>> 0).toString(36);
}
