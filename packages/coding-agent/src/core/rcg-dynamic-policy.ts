import type {
	SpeculativeSwarmDynamicAction,
	SpeculativeSwarmDynamicPlanInput,
	SpeculativeSwarmDynamicPolicy,
	SpeculativeSwarmDynamicRoute,
	SpeculativeSwarmDynamicRouteInput,
	SpeculativeSwarmDynamicSelection,
	SpeculativeSwarmDynamicSelectionInput,
} from "./speculative-swarm.ts";

interface PlanResponse {
	actions: SpeculativeSwarmDynamicAction[];
	modelStep?: number;
	latencyMs?: number;
}

interface SelectionResponse extends SpeculativeSwarmDynamicSelection {
	modelStep?: number;
	latencyMs?: number;
}

export class HttpRcgDynamicPolicy implements SpeculativeSwarmDynamicPolicy {
	private readonly baseUrl: string;
	private readonly timeoutMs: number;

	constructor(baseUrl: string, timeoutMs = 30_000) {
		this.baseUrl = baseUrl.replace(/\/+$/, "");
		this.timeoutMs = timeoutMs;
	}

	async plan(input: SpeculativeSwarmDynamicPlanInput): Promise<SpeculativeSwarmDynamicAction[]> {
		const response = await this.post<PlanResponse>("/plan", input);
		return response.actions;
	}

	async select(input: SpeculativeSwarmDynamicSelectionInput): Promise<SpeculativeSwarmDynamicSelection> {
		return this.post<SelectionResponse>("/select", input);
	}

	async route(input: SpeculativeSwarmDynamicRouteInput): Promise<SpeculativeSwarmDynamicRoute> {
		return this.post<SpeculativeSwarmDynamicRoute>("/route", input);
	}

	async health(): Promise<Record<string, unknown>> {
		return this.post<Record<string, unknown>>("/health", {});
	}

	private async post<T>(path: string, payload: unknown): Promise<T> {
		const response = await fetch(`${this.baseUrl}${path}`, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify(payload),
			signal: AbortSignal.timeout(this.timeoutMs),
		});
		if (!response.ok) {
			const body = await response.text();
			throw new Error(`RCG policy ${path} failed (${response.status}): ${body.slice(0, 500)}`);
		}
		return (await response.json()) as T;
	}
}
