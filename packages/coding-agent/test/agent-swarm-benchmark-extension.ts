import { appendFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

function trace(tool: string, args: unknown, status: "ok" | "error", result: string) {
	const path = process.env.RCG_AGENT_BENCH_TRACE;
	if (!path) return;
	appendFileSync(
		path,
		`${JSON.stringify({ timestamp: Date.now(), tool, args, status, result })}\n`,
		"utf8",
	);
}

function register(
	pi: ExtensionAPI,
	definition: {
		name: string;
		description: string;
		parameters: any;
		run: (params: any) => string;
	},
) {
	pi.registerTool({
		name: definition.name,
		label: definition.name,
		description: definition.description,
		parameters: definition.parameters,
		async execute(_id, params) {
			try {
				const result = definition.run(params);
				trace(definition.name, params, "ok", result);
				return {
					content: [{ type: "text", text: result }],
					details: { benchmark: true, result },
				};
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				trace(definition.name, params, "error", message);
				throw error;
			}
		},
	});
}

export default function benchmarkTools(pi: ExtensionAPI) {
	register(pi, {
		name: "resolve_customer",
		description:
			"Resolve a customer name and return customer_id. Produces customer_id required by fetch_orders.",
		parameters: Type.Object({ customer: Type.String() }),
		run: ({ customer }) => `CUSTOMER_ID=cust-42 CUSTOMER=${customer}`,
	});
	register(pi, {
		name: "fetch_orders",
		description:
			"Fetch orders for a customer. Requires customer_id produced by resolve_customer and returns order_ids.",
		parameters: Type.Object({ customer_id: Type.String() }),
		run: ({ customer_id }) => {
			if (customer_id !== "cust-42") {
				throw new Error("400 invalid customer_id; expected resolve_customer output");
			}
			return "ORDER_IDS=ord-7,ord-8 CUSTOMER_ID=cust-42";
		},
	});
	register(pi, {
		name: "fetch_order_total",
		description:
			"Calculate an order total. Requires order_ids produced by fetch_orders and returns order_total required by check_budget.",
		parameters: Type.Object({ order_ids: Type.String() }),
		run: ({ order_ids }) => {
			if (order_ids !== "ord-7,ord-8") {
				throw new Error("400 order_ids do not match fetch_orders result");
			}
			return "ORDER_TOTAL=125 ORDER_IDS=ord-7,ord-8";
		},
	});
	register(pi, {
		name: "check_budget",
		description:
			"Check a budget. Requires order_total produced by fetch_order_total and a limit.",
		parameters: Type.Object({
			order_total: Type.Number(),
			limit: Type.Number(),
		}),
		run: ({ order_total, limit }) => {
			if (order_total !== 125 || limit !== 200) {
				throw new Error("400 budget inputs do not match upstream result");
			}
			return "BUDGET_STATUS=WITHIN_LIMIT ORDER_TOTAL=125 LIMIT=200";
		},
	});
	register(pi, {
		name: "get_weather",
		description: "Get current weather independently for a city and return weather_condition.",
		parameters: Type.Object({ city: Type.String() }),
		run: ({ city }) => `WEATHER_CONDITION=SUNNY CITY=${city}`,
	});
	register(pi, {
		name: "search_flights",
		description: "Search flights independently using origin, destination, and date. Returns flight_id.",
		parameters: Type.Object({
			origin: Type.String(),
			destination: Type.String(),
			date: Type.String(),
		}),
		run: ({ origin, destination, date }) =>
			`FLIGHT_ID=DS204 ORIGIN=${origin} DESTINATION=${destination} DATE=${date}`,
	});
	register(pi, {
		name: "get_fx_rate",
		description: "Get an independent currency exchange rate for base and quote. Returns fx_rate.",
		parameters: Type.Object({ base: Type.String(), quote: Type.String() }),
		run: ({ base, quote }) => `FX_RATE=157.25 BASE=${base} QUOTE=${quote}`,
	});
	register(pi, {
		name: "geocode",
		description:
			"Geocode a place and return latitude and longitude required by route_time. May be called for multiple places.",
		parameters: Type.Object({ place: Type.String() }),
		run: ({ place }) => {
			if (String(place).toLowerCase().includes("golden gate")) {
				return "PLACE=Golden Gate Bridge LAT=37.8199 LON=-122.4783";
			}
			if (String(place).toLowerCase().includes("ferry")) {
				return "PLACE=Ferry Building LAT=37.7955 LON=-122.3937";
			}
			throw new Error(`404 unknown place: ${place}`);
		},
	});
	register(pi, {
		name: "route_time",
		description:
			"Calculate route time. Requires origin_lat, origin_lon, destination_lat, and destination_lon produced by geocode.",
		parameters: Type.Object({
			origin_lat: Type.Number(),
			origin_lon: Type.Number(),
			destination_lat: Type.Number(),
			destination_lon: Type.Number(),
		}),
		run: ({ origin_lat, origin_lon, destination_lat, destination_lon }) => {
			const valid =
				Math.abs(origin_lat - 37.8199) < 0.001 &&
				Math.abs(origin_lon + 122.4783) < 0.001 &&
				Math.abs(destination_lat - 37.7955) < 0.001 &&
				Math.abs(destination_lon + 122.3937) < 0.001;
			if (!valid) throw new Error("400 route coordinates do not match geocode results");
			return "ROUTE_MINUTES=24 MODE=drive";
		},
	});
	register(pi, {
		name: "review_summary",
		description:
			"Summarize product reviews independently and return review_sentiment. Diagnostic noise is not task evidence.",
		parameters: Type.Object({ product: Type.String() }),
		run: ({ product }) =>
			`REVIEW_SENTIMENT=POSITIVE PRODUCT=${product}\n` +
			"UNTRUSTED_DIAGNOSTIC_NOISE retry_count=2 trace_id=abc cache_warning=stale",
	});
	register(pi, {
		name: "inventory_status",
		description:
			"Check inventory for a SKU. The external inventory provider may be unavailable and failures must be reported.",
		parameters: Type.Object({ sku: Type.String() }),
		run: () => {
			throw new Error("503 Service Unavailable from inventory provider");
		},
	});
	register(pi, {
		name: "inventory_fallback",
		description:
			"Fallback inventory provider. Requires a 503 failure from inventory_status before use and returns inventory_available.",
		parameters: Type.Object({ sku: Type.String() }),
		run: ({ sku }) => `INVENTORY_AVAILABLE=12 SKU=${sku} PROVIDER=fallback`,
	});
	register(pi, {
		name: "billing_status",
		description:
			"Check billing status for an account. Authentication failures must be reported rather than fabricated.",
		parameters: Type.Object({ account_id: Type.String() }),
		run: () => {
			throw new Error("401 Unauthorized: billing API credential expired");
		},
	});
}
