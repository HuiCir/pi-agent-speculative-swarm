#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, join, relative } from "node:path";

function parseArgs(argv) {
	const options = {};
	for (let index = 0; index < argv.length; index++) {
		const arg = argv[index];
		if (arg === "--data-root") options.dataRoot = argv[++index];
		else if (arg === "--report") options.report = argv[++index];
		else if (arg === "--normalized") options.normalized = argv[++index];
		else if (arg === "--help" || arg === "-h") options.help = true;
		else throw new Error(`Unknown argument: ${arg}`);
	}
	return options;
}

function usage() {
	return `Usage:
  node scripts/validate-traject-swarm-data.mjs \\
    --data-root /path/to/TRAJECT-Bench/public_data \\
    --report /path/to/validation.json \\
    --normalized /path/to/swarm_training.jsonl
`;
}

function readJson(file) {
	return JSON.parse(readFileSync(file, "utf8"));
}

function listDirectories(root) {
	return readdirSync(root)
		.map((name) => join(root, name))
		.filter((path) => statSync(path).isDirectory())
		.sort();
}

function getTools(row) {
	const tools = row["tool list"] ?? row.tool_list;
	return Array.isArray(tools) ? tools : [];
}

function hash(value) {
	return createHash("sha1").update(value).digest("hex").slice(0, 16);
}

function statusOf(tool) {
	if (tool.execution_status === "success" || tool.execution_status === "failed") {
		return tool.execution_status;
	}
	const output = String(tool.executed_output ?? "").trim();
	if (!output) return "unknown";
	return /^error\b|failed after/i.test(output) ? "failed" : "success";
}

function parameterList(tool, key) {
	const values = tool[key];
	if (!Array.isArray(values)) return [];
	return values.map((parameter) => ({
		name: String(parameter?.name ?? ""),
		value: parameter?.value,
	}));
}

function dependencyMentioned(parameter, tool) {
	if (!parameter) return false;
	const haystack = [
		tool["tool description"],
		tool.sequence_step?.description,
		tool.adapt_parameter,
		tool.adapt_constraint,
		tool.adapt_reason,
		...parameterList(tool, "required parameters").map((item) => item.name),
		...parameterList(tool, "optional parameters").map((item) => item.name),
	]
		.filter(Boolean)
		.join(" ")
		.toLowerCase();
	return haystack.includes(String(parameter).toLowerCase());
}

function normalizeBranch(tool, index, totalTools, trajectoryType, previousTool) {
	const step = tool.sequence_step?.step_number ?? (trajectoryType === "sequential" ? index + 1 : undefined);
	const annotatedProducedParameter = String(tool.sequence_step?.param_for_next_tool ?? "").trim();
	const inferredProducedParameter =
		trajectoryType === "sequential" && index < totalTools - 1
			? `__step_${index + 1}_completion_or_output__`
			: "";
	const producedParameter = annotatedProducedParameter || inferredProducedParameter;
	const annotatedDependencyParameter =
		trajectoryType === "sequential"
			? String(previousTool?.sequence_step?.param_for_next_tool ?? "").trim()
			: "";
	const inferredDependencyParameter =
		trajectoryType === "sequential" && index > 0 ? `__step_${index}_completion_or_output__` : "";
	const dependencyParameter = annotatedDependencyParameter || inferredDependencyParameter;
	return {
		branch_id: `${trajectoryType === "parallel" ? "p" : "s"}${index + 1}`,
		preallocated_stage: 1,
		earliest_commit_stage: trajectoryType === "parallel" ? 1 : step,
		sequence_step: step ?? null,
		tool_name: String(tool["tool name"] ?? tool.tool_name ?? ""),
		tool_description: String(tool["tool description"] ?? tool.tool_description ?? ""),
		required_parameters: parameterList(tool, "required parameters"),
		optional_parameters: parameterList(tool, "optional parameters"),
		execution_status: statusOf(tool),
		executed_output: String(tool.executed_output ?? ""),
		depends_on:
			dependencyParameter && previousTool
				? [
						{
							source_branch_id: `s${index}`,
							parameter: dependencyParameter,
							annotation_source: annotatedDependencyParameter ? "dataset" : "inferred_sequence_order",
							explicitly_referenced:
								Boolean(annotatedDependencyParameter) &&
								dependencyMentioned(annotatedDependencyParameter, tool),
						},
					]
				: [],
		produces: producedParameter
			? [
					{
						parameter: producedParameter,
						annotation_source: annotatedProducedParameter ? "dataset" : "inferred_sequence_order",
					},
				]
			: [],
		adaptation: {
			parameter: tool.adapt_parameter ?? null,
			constraint: tool.adapt_constraint ?? null,
			reason: tool.adapt_reason ?? null,
		},
		controller_action:
			trajectoryType === "parallel"
				? "execute_independently"
				: index === 0
					? "execute_and_publish_dependency"
					: "preexecute_if_safe_then_resume_same_session_on_dependency",
	};
}

function normalizeRow(row, sourceFile, index, dataRoot, trajectoryType) {
	const tools = getTools(row);
	const domain = String(row.domain ?? tools[0]?.["domain name"] ?? basename(dirname(sourceFile)));
	const query = String(row.query ?? "");
	const branches = tools.map((tool, toolIndex) =>
		normalizeBranch(tool, toolIndex, tools.length, trajectoryType, tools[toolIndex - 1]),
	);
	const signature = JSON.stringify({
		trajectoryType,
		domain,
		query,
		tools: branches.map((branch) => ({
			name: branch.tool_name,
			required: branch.required_parameters,
			optional: branch.optional_parameters,
		})),
	});
	return {
		task_id: `${trajectoryType}-${domain}-${hash(signature)}`,
		source: relative(dataRoot, sourceFile),
		source_index: index,
		domain,
		trajectory_type: trajectoryType,
		difficulty:
			trajectoryType === "parallel"
				? basename(sourceFile).replace("_ver.json", "")
				: basename(sourceFile) === "simple_ver.json"
					? "simple"
					: "trajectory",
		query,
		final_answer: row.final_answer ?? null,
		sequence_name: row.sequence_name ?? null,
		sequence_description: row.sequence_description ?? null,
		executable: row.executable ?? null,
		controller_target: {
			preallocate_all_branches_at_turn_1: true,
			release_strategy: trajectoryType === "parallel" ? "parallel_barrier" : "cooperative_wavefront",
			resume_same_subagent_session: trajectoryType === "sequential",
			required_branch_slots: branches.length,
			requires_dynamic_slots_beyond_default_four: branches.length > 4,
			coherence_label_policy:
				"derive_from_replay_quality_and_dependency_correctness; do_not_use_tool_status_alone",
		},
		branches,
		_signature: signature,
	};
}

function loadRows(dataRoot) {
	const loaded = [];
	const parallelRoot = join(dataRoot, "parallel");
	for (const domainPath of listDirectories(parallelRoot)) {
		for (const fileName of ["simple_ver.json", "hard_ver.json"]) {
			const file = join(domainPath, fileName);
			if (!existsSync(file)) continue;
			const rows = readJson(file);
			rows.forEach((row, index) => loaded.push(normalizeRow(row, file, index, dataRoot, "parallel")));
		}
	}

	const sequentialRoot = join(dataRoot, "sequential");
	for (const domainPath of listDirectories(sequentialRoot)) {
		for (const fileName of ["traj_query.json", "simple_ver.json"]) {
			const file = join(domainPath, fileName);
			if (!existsSync(file)) continue;
			const rows = readJson(file);
			if (!Array.isArray(rows) || !rows.every((row) => typeof row.query === "string")) continue;
			rows.forEach((row, index) => loaded.push(normalizeRow(row, file, index, dataRoot, "sequential")));
		}
	}
	return loaded;
}

function increment(record, key, amount = 1) {
	record[key] = (record[key] ?? 0) + amount;
}

function validate(rows) {
	const issues = [];
	const byType = {};
	const byDomain = {};
	const bySource = {};
	const sequenceLengths = {};
	const stepStatus = {};
	let branchCount = 0;
	let dependencyEdges = 0;
	let explicitDependencyEdges = 0;
	let datasetAnnotatedDependencyEdges = 0;
	let inferredDependencyEdges = 0;
	let adaptedConsumers = 0;
	let executableRowsWithFailedSteps = 0;
	let structurallyValidRows = 0;
	let sequentialRowsFittingFourSlots = 0;
	let sequentialRowsRequiringDynamicSlots = 0;
	let maxSequentialBranchSlots = 0;

	for (const row of rows) {
		increment(byType, row.trajectory_type);
		increment(byDomain, `${row.trajectory_type}/${row.domain}`);
		increment(bySource, row.source);
		increment(sequenceLengths, `${row.trajectory_type}/${row.branches.length}`);
		branchCount += row.branches.length;
		if (row.trajectory_type === "sequential") {
			maxSequentialBranchSlots = Math.max(maxSequentialBranchSlots, row.branches.length);
			if (row.branches.length <= 4) sequentialRowsFittingFourSlots++;
			else sequentialRowsRequiringDynamicSlots++;
		}

		const rowIssues = [];
		if (!row.query.trim()) rowIssues.push("missing query");
		if (row.branches.length === 0) rowIssues.push("missing tools");
		if (row.final_answer === null) rowIssues.push("missing final answer");

		for (let index = 0; index < row.branches.length; index++) {
			const branch = row.branches[index];
			increment(stepStatus, `${row.trajectory_type}/${branch.execution_status}`);
			if (!branch.tool_name) rowIssues.push(`step ${index + 1} missing tool name`);
			if (row.trajectory_type === "sequential") {
				if (branch.sequence_step !== index + 1) {
					rowIssues.push(`step order mismatch at ${index + 1}`);
				}
				if (index > 0) {
					dependencyEdges += branch.depends_on.length;
					explicitDependencyEdges += branch.depends_on.filter((edge) => edge.explicitly_referenced).length;
					datasetAnnotatedDependencyEdges += branch.depends_on.filter(
						(edge) => edge.annotation_source === "dataset",
					).length;
					inferredDependencyEdges += branch.depends_on.filter(
						(edge) => edge.annotation_source === "inferred_sequence_order",
					).length;
					if (branch.adaptation.parameter || branch.adaptation.constraint || branch.adaptation.reason) {
						adaptedConsumers++;
					}
					if (branch.depends_on.length !== 1 || !branch.depends_on[0].parameter) {
						rowIssues.push(`step ${index + 1} missing dependency edge`);
					}
				}
				if (index < row.branches.length - 1 && branch.produces.length !== 1) {
					rowIssues.push(`step ${index + 1} missing produced parameter`);
				}
			}
		}

		if (
			row.trajectory_type === "sequential" &&
			row.executable === true &&
			row.branches.some((branch) => branch.execution_status === "failed")
		) {
			executableRowsWithFailedSteps++;
		}
		if (rowIssues.length === 0) {
			structurallyValidRows++;
		} else if (issues.length < 100) {
			issues.push({ task_id: row.task_id, source: row.source, source_index: row.source_index, issues: rowIssues });
		}
	}

	const unique = new Map();
	const duplicates = [];
	for (const row of rows) {
		const existing = unique.get(row._signature);
		if (existing) {
			duplicates.push({
				task_id: row.task_id,
				source: row.source,
				source_index: row.source_index,
				duplicate_of: { source: existing.source, source_index: existing.source_index },
			});
		} else {
			unique.set(row._signature, row);
		}
	}

	return {
		uniqueRows: Array.from(unique.values()),
		report: {
			schema_version: 1,
			raw_rows: rows.length,
			unique_training_rows: unique.size,
			duplicate_rows: duplicates.length,
			total_preallocated_branches: branchCount,
			structurally_valid_rows: structurallyValidRows,
			structurally_invalid_rows: rows.length - structurallyValidRows,
			counts_by_type: byType,
			counts_by_domain: byDomain,
			counts_by_source: bySource,
			sequence_length_distribution: sequenceLengths,
			step_execution_status: stepStatus,
			sequential_dependency_edges: dependencyEdges,
			sequential_dependency_edges_dataset_annotated: datasetAnnotatedDependencyEdges,
			sequential_dependency_edges_inferred_from_order: inferredDependencyEdges,
			sequential_dependency_edges_explicitly_referenced: explicitDependencyEdges,
			sequential_dependency_explicit_rate:
				dependencyEdges === 0 ? null : explicitDependencyEdges / dependencyEdges,
			sequential_adapted_consumers: adaptedConsumers,
			executable_rows_with_failed_steps: executableRowsWithFailedSteps,
			runtime_capacity_analysis: {
				default_branch_slots: 4,
				max_sequential_branch_slots_required: maxSequentialBranchSlots,
				sequential_rows_fitting_default_four_slots: sequentialRowsFittingFourSlots,
				sequential_rows_requiring_dynamic_slots_or_windowing: sequentialRowsRequiringDynamicSlots,
			},
			duplicate_examples: duplicates.slice(0, 30),
			structural_issue_examples: issues,
			training_guidance: [
				"Use every unique parallel and sequential trajectory; retain source metadata for domain-balanced sampling.",
				"Preallocate all sequential branches in stage 1, but supervise commit order with dependency edges.",
				"Resume the same subagent session after upstream outputs arrive instead of regenerating the branch.",
				"Train the controller with up to 10 dynamic sequential slots; fixed four-slot deployments must use dependency-preserving trajectory windows.",
				"Down-weight parameter-name supervision on inferred sequence-order edges; they preserve control flow but not a verified field name.",
				"Treat execution_status as observed tool evidence, not as the final coherence label.",
				"Derive coherence from replayed task quality, dependency correctness, non-duplication, and final answer coverage.",
			],
		},
	};
}

function stripInternal(row) {
	const { _signature, ...clean } = row;
	return clean;
}

const options = parseArgs(process.argv.slice(2));
if (options.help) {
	console.log(usage());
	process.exit(0);
}
if (!options.dataRoot || !options.report || !options.normalized) {
	console.error(usage());
	process.exit(2);
}
if (!existsSync(options.dataRoot)) {
	throw new Error(`Data root does not exist: ${options.dataRoot}`);
}

const rows = loadRows(options.dataRoot);
const { uniqueRows, report } = validate(rows);
mkdirSync(dirname(options.report), { recursive: true });
mkdirSync(dirname(options.normalized), { recursive: true });
writeFileSync(options.report, `${JSON.stringify(report, null, 2)}\n`, "utf8");
writeFileSync(options.normalized, `${uniqueRows.map((row) => JSON.stringify(stripInternal(row))).join("\n")}\n`, "utf8");

console.log(JSON.stringify({ report: options.report, normalized: options.normalized, ...report }, null, 2));
if (report.structurally_invalid_rows > 0) {
	process.exitCode = 1;
}
