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
import { createSpeculativeSwarmController } from "../src/core/speculative-swarm.ts";

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

function assistant(content: AssistantMessage["content"], stopReason: AssistantMessage["stopReason"] = "stop") {
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
	} satisfies AssistantMessage;
}

function tool(name: string, result: string): AgentTool {
	return {
		name,
		label: name,
		description: name === "weather" ? "Get weather for a city" : "Search flights between cities",
		parameters: Type.Object({ city: Type.Optional(Type.String()) }),
		async execute() {
			return { content: [{ type: "text", text: result }], details: { result } };
		},
	};
}

const calls = new Map<string, number>();
const plannedTools = new Map<string, string>();
const streamFn = (_model: Model<any>, _context: { messages: Message[] }, options?: { sessionId?: string }) => {
	const session = options?.sessionId ?? "unknown";
	const invocation = (calls.get(session) ?? 0) + 1;
	calls.set(session, invocation);
	const stream = new MockStream();
	queueMicrotask(() => {
		let message: AssistantMessage;
		if (session.includes("Turn0OrchestratorBrief")) {
			message = assistant([{ type: "text", text: "BRANCH_PROTOTYPES:\n- branch_id: all\n  vote_for_when: tool evidence\n  reject_when: error\n  must_verify: result" }]);
		} else {
			const briefId = session.match(/rcg-action-\d+/)?.[0] ?? "";
			const toolName = plannedTools.get(briefId) ?? "flights";
			const isWeather = toolName === "weather";
			const result = isWeather ? "WEATHER=SUNNY" : "FLIGHT=AF123";
			if (invocation === 1) {
				message = assistant(
					[
						{
							type: "toolCall",
							id: `${toolName}-${session}`,
							name: toolName,
							arguments: { city: isWeather ? "London" : "Paris" },
						},
					],
					"toolUse",
				);
			} else {
				message = assistant([{ type: "text", text: result }]);
			}
		}
		stream.push({ type: "done", reason: message.stopReason === "toolUse" ? "toolUse" : "stop", message });
	});
	return stream;
};

const httpPolicy = new HttpRcgDynamicPolicy("http://127.0.0.1:8765", 60_000);
const policy = {
	async plan(input: Parameters<HttpRcgDynamicPolicy["plan"]>[0]) {
		const actions = await httpPolicy.plan(input);
		for (const action of actions) {
			if (action.toolName) plannedTools.set(action.id, action.toolName);
		}
		return actions;
	},
	select: httpPolicy.select.bind(httpPolicy),
};
const controller = createSpeculativeSwarmController({
	enabled: true,
	model,
	agents: ["A", "B"],
	rounds: 1,
	maxSubagentTurns: 2,
	timeoutMs: 20_000,
	toolPolicy: "readonly",
	dynamicPolicy: policy,
});
if (!controller) throw new Error("controller was not created");

const userMessage: AgentMessage = {
	role: "user",
	content: [{ type: "text", text: "Fetch weather for London and find flights to Paris." }],
	timestamp: Date.now(),
};
const started = Date.now();
const enriched = await controller.prepareContext({
	context: {
		systemPrompt: "Integration test",
		tools: [tool("weather", "WEATHER=SUNNY"), tool("flights", "FLIGHT=AF123")],
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
const mainModelCalls = 1;
if (!memory.includes("WEATHER=SUNNY") || !memory.includes("FLIGHT=AF123")) {
	throw new Error(`RCG swarm total draft is incomplete: ${memory.slice(-2000)}`);
}
console.log(JSON.stringify({
	ok: true,
	checkpointStep: (await httpPolicy.health()).modelStep,
	prepareMs,
	mainModelCalls,
	subagentSessions: Array.from(calls.entries()),
	hasWeather: memory.includes("WEATHER=SUNNY"),
	hasFlights: memory.includes("FLIGHT=AF123"),
}, null, 2));
