import type { AgentMessage, AgentTool } from "@earendil-works/pi-agent-core";
import {
	type AssistantMessage,
	type AssistantMessageEvent,
	EventStream,
	type Message,
	type Model,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { HttpRcgDynamicPolicy } from "../src/core/rcg-dynamic-policy.ts";
import {
	createSpeculativeSwarmController,
	type SpeculativeSwarmDynamicPlanInput,
} from "../src/core/speculative-swarm.ts";

class MockStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event");
			},
		);
	}
}

const model: Model<"openai-responses"> = {
	id: "fault-injection-mock",
	name: "fault-injection-mock",
	api: "openai-responses",
	provider: "openai",
	baseUrl: "https://example.invalid",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 8192,
	maxTokens: 2048,
};

function assistant(content: AssistantMessage["content"], stopReason: AssistantMessage["stopReason"] = "stop") {
	return {
		role: "assistant",
		content,
		api: "openai-responses",
		provider: "openai",
		model: model.id,
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
	} satisfies AssistantMessage;
}

const toolExecutions = new Map<string, number>();
function count(name: string) {
	toolExecutions.set(name, (toolExecutions.get(name) ?? 0) + 1);
}

function tools(): AgentTool[] {
	return [
		{
			name: "lookup_account",
			label: "lookup_account",
			description: "Look up a customer and produce the account_id required by fetch_orders.",
			parameters: Type.Object({ customer: Type.String() }),
			async execute(_id, args) {
				count("lookup_account");
				return {
					content: [{ type: "text", text: `ACCOUNT_ID=acct-42 CUSTOMER=${args.customer}` }],
					details: { accountId: "acct-42" },
				};
			},
		},
		{
			name: "fetch_orders",
			label: "fetch_orders",
			description: "Fetch orders. Requires account_id produced by lookup_account.",
			parameters: Type.Object({ account_id: Type.String() }),
			async execute(_id, args) {
				count("fetch_orders");
				if (args.account_id !== "acct-42") {
					throw new Error("400 invalid account_id; requires lookup_account result");
				}
				return {
					content: [{ type: "text", text: "ORDERS=order-7,order-8 ACCOUNT_ID=acct-42" }],
					details: { orders: ["order-7", "order-8"] },
				};
			},
		},
		{
			name: "weather",
			label: "weather",
			description: "Fetch current weather independently for a city.",
			parameters: Type.Object({ city: Type.String() }),
			async execute(_id, args) {
				count("weather");
				return {
					content: [
						{
							type: "text",
							text:
								`WEATHER=SUNNY CITY=${args.city}\n` +
								"UNTRUSTED_DIAGNOSTIC_NOISE retry_count=2 trace_id=abc cache_warning=stale",
						},
					],
					details: { weather: "SUNNY" },
				};
			},
		},
		{
			name: "broken_api",
			label: "broken_api",
			description: "Probe a requested external status API that may be unavailable.",
			parameters: Type.Object({ resource: Type.String() }),
			async execute() {
				count("broken_api");
				throw new Error("503 Service Unavailable");
			},
		},
	];
}

const calls = new Map<string, number>();
const plannedTools = new Map<string, string>();
let sawDependencyEvidence = false;

const streamFn = (_model: Model<any>, context: { messages: Message[] }, options?: { sessionId?: string }) => {
	const session = options?.sessionId ?? "unknown";
	const invocation = (calls.get(session) ?? 0) + 1;
	calls.set(session, invocation);
	const transcript = JSON.stringify(context.messages);
	const stream = new MockStream();
	queueMicrotask(() => {
		let message: AssistantMessage;
		if (session.includes("Turn0OrchestratorBrief")) {
			message = assistant([
				{
					type: "text",
					text:
						"BRANCH_PROTOTYPES:\n- branch_id: all\n  vote_for_when: verified local tool result\n" +
						"  reject_when: API error or wrong dependency\n  must_verify: tool outcome",
				},
			]);
		} else {
			const briefId = session.match(/rcg-action-\d+/)?.[0] ?? "";
			const toolName = plannedTools.get(briefId);
			if (!toolName) throw new Error(`No planned tool for ${session}`);
			if (toolName === "lookup_account") {
				message =
					invocation === 1
						? assistant(
								[
									{
										type: "toolCall",
										id: `lookup-${session}`,
										name: toolName,
										arguments: { customer: "Alice" },
									},
								],
								"toolUse",
							)
						: assistant([{ type: "text", text: "ACCOUNT_ID=acct-42 verified for Alice" }]);
			} else if (toolName === "fetch_orders") {
				sawDependencyEvidence ||= transcript.includes("ACCOUNT_ID=acct-42");
				if (invocation <= 2) {
					message = assistant(
						[
							{
								type: "toolCall",
								id: `orders-${session}-${invocation}`,
								name: toolName,
								arguments: {
									account_id: sawDependencyEvidence ? "acct-42" : "MISSING",
								},
							},
						],
						"toolUse",
					);
				} else {
					message = assistant([
						{ type: "text", text: "ORDERS=order-7,order-8 using ACCOUNT_ID=acct-42" },
					]);
				}
			} else if (toolName === "weather") {
				message =
					invocation === 1
						? assistant(
								[
									{
										type: "toolCall",
										id: `weather-${session}`,
										name: toolName,
										arguments: { city: "London" },
									},
								],
								"toolUse",
							)
						: assistant([
								{
									type: "text",
									text:
										"WEATHER=SUNNY CITY=London; ignored diagnostic cache warning as non-task noise",
								},
							]);
			} else {
				message = assistant(
					[
						{
							type: "toolCall",
							id: `broken-${session}-${invocation}`,
							name: toolName,
							arguments: { resource: "requested-status" },
						},
					],
					"toolUse",
				);
			}
		}
		stream.push({
			type: "done",
			reason: message.stopReason === "toolUse" ? "toolUse" : "stop",
			message,
		});
	});
	return stream;
};

const httpPolicy = new HttpRcgDynamicPolicy("http://127.0.0.1:8765", 60_000);
const routeTrace: unknown[] = [];
const selectionTrace: unknown[] = [];
const policy = {
	async plan(input: SpeculativeSwarmDynamicPlanInput) {
		const actions = await httpPolicy.plan(input);
		for (const action of actions) {
			if (action.id && action.toolName) plannedTools.set(action.id, action.toolName);
		}
		return actions;
	},
	async route(input: Parameters<HttpRcgDynamicPolicy["route"]>[0]) {
		const routed = await httpPolicy.route(input);
		routeTrace.push(routed);
		return routed;
	},
	async select(input: Parameters<HttpRcgDynamicPolicy["select"]>[0]) {
		const selected = await httpPolicy.select(input);
		selectionTrace.push(selected);
		return selected;
	},
};

const controller = createSpeculativeSwarmController({
	enabled: true,
	model,
	agents: ["A", "B", "C", "D"],
	rounds: 1,
	maxSubagentTurns: 3,
	timeoutMs: 20_000,
	toolPolicy: "readonly",
	dynamicPolicy: policy,
});
if (!controller) throw new Error("controller was not created");

const userMessage: AgentMessage = {
	role: "user",
	content: [
		{
			type: "text",
			text:
				"First look up Alice's account id, then use that result to fetch her orders. " +
				"Also fetch London weather independently and probe the requested external status API.",
		},
	],
	timestamp: Date.now(),
};

const started = Date.now();
const enriched = await controller.prepareContext({
	context: {
		systemPrompt: "Fault-injection integration test",
		tools: tools(),
		messages: [userMessage],
	},
	model,
	thinkingLevel: "off",
	convertToLlm: (messages) => messages as Message[],
	streamFn,
	requestIndex: 0,
});
const prepareMs = Date.now() - started;
const memory = JSON.stringify(enriched?.messages ?? []);
const finalSelection = selectionTrace.at(-1) as
	| {
			scores?: Array<{ briefId: string; retain: boolean }>;
		}
	| undefined;
const retainedTools = new Set(
	(finalSelection?.scores ?? [])
		.filter((score) => score.retain)
		.map((score) => plannedTools.get(score.briefId)),
);

for (const expected of ["ACCOUNT_ID=acct-42", "ORDERS=order-7,order-8", "WEATHER=SUNNY"]) {
	if (!memory.includes(expected)) throw new Error(`Missing retained evidence: ${expected}`);
}
if (!sawDependencyEvidence) throw new Error("fetch_orders never received routed account evidence");
for (const expected of ["lookup_account", "fetch_orders", "weather"]) {
	if (!retainedTools.has(expected)) throw new Error(`RCG did not retain valid path: ${expected}`);
}
if (retainedTools.has("broken_api")) {
	throw new Error("broken API branch was retained");
}

console.log(
	JSON.stringify(
		{
			ok: true,
			checkpointStep: (await httpPolicy.health()).modelStep,
			prepareMs,
			mainModelCalls: 1,
			sawDependencyEvidence,
			plannedTools: Object.fromEntries(plannedTools),
			toolExecutions: Object.fromEntries(toolExecutions),
			subagentCalls: Object.fromEntries(calls),
			routeTrace,
			selectionTrace,
			retainedTools: Array.from(retainedTools),
			retained: {
				account: retainedTools.has("lookup_account"),
				orders: retainedTools.has("fetch_orders"),
				weather: retainedTools.has("weather"),
				brokenSuppressed: !retainedTools.has("broken_api"),
			},
		},
		null,
		2,
	),
);
