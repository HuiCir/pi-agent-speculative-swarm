import unittest

from build_failure_aware_training_data import annotate_task


def branch(branch_id, status, depends_on=None):
    return {
        "branch_id": branch_id,
        "tool_name": f"tool-{branch_id}",
        "tool_description": f"objective-{branch_id}",
        "depends_on": [
            {
                "source_branch_id": source,
                "parameter": "value",
                "annotation_source": "dataset",
            }
            for source in (depends_on or [])
        ],
        "live_replay": {"status": status},
    }


class FailureAwareDataTest(unittest.TestCase):
    def test_parallel_failure_makes_task_unsolvable(self):
        row = annotate_task(
            {
                "task_id": "parallel-test",
                "branches": [
                    branch("p1", "live_valid"),
                    branch("p2", "auth_or_quota"),
                    branch("p3", "live_valid"),
                ],
            }
        )
        self.assertEqual(row["task_solvability_target"], 0.0)
        self.assertEqual(
            row["task_outcome"]["root_failures"][0]["branch_id"], "p2"
        )
        self.assertIn(
            "TASK_STATUS: UNSOLVABLE_CURRENT_PLAN",
            row["failure_report_target"],
        )
        self.assertIn("auth_or_quota", row["failure_report_target"])

    def test_upstream_failure_blocks_valid_downstream_calls(self):
        row = annotate_task(
            {
                "task_id": "sequential-test",
                "branches": [
                    branch("s1", "live_valid"),
                    branch("s2", "timeout", ["s1"]),
                    branch("s3", "live_valid", ["s2"]),
                ],
            }
        )
        by_id = {item["branch_id"]: item for item in row["branches"]}
        self.assertEqual(by_id["s2"]["failure_source_target"], 1.0)
        self.assertEqual(by_id["s3"]["execution_coherence_target"], 1.0)
        self.assertEqual(by_id["s3"]["effective_coherence_target"], 0.0)
        self.assertEqual(by_id["s3"]["failure_source_target"], 0.0)
        self.assertEqual(by_id["s3"]["failure_type"], "blocked_dependency")
        self.assertEqual(by_id["s3"]["blocked_by"], ["s2"])
        self.assertEqual(
            row["task_outcome"]["status"], "unsolvable_current_plan"
        )
        self.assertEqual(
            row["task_outcome"]["root_failures"][0]["blocked_descendants"],
            ["s3"],
        )

    def test_all_valid_task_remains_solvable(self):
        row = annotate_task(
            {
                "task_id": "valid-test",
                "branches": [
                    branch("s1", "live_valid"),
                    branch("s2", "live_valid", ["s1"]),
                ],
            }
        )
        self.assertEqual(row["task_solvability_target"], 1.0)
        self.assertEqual(row["task_outcome"]["root_failures"], [])
        self.assertIn("TASK_STATUS: SOLVABLE", row["failure_report_target"])
        self.assertEqual(row["takeover_target"], 1.0)
        self.assertEqual(row["main_assembly_mode"], "direct_branch")
        self.assertTrue(
            all(
                item["continuation_target"] == 0.0
                for item in row["branches"]
            )
        )

    def test_parallel_valid_paths_form_one_takeover_set(self):
        row = annotate_task(
            {
                "task_id": "parallel-valid-test",
                "branches": [
                    branch("p1", "live_valid"),
                    branch("p2", "live_valid"),
                    branch("p3", "live_valid"),
                ],
            }
        )
        self.assertEqual(row["takeover_target"], 1.0)
        self.assertEqual(
            row["main_assembly_mode"], "coherent_set_takeover"
        )
        self.assertTrue(
            all(
                item["main_candidate_target"] == 1.0
                for item in row["branches"]
            )
        )

    def test_retryable_failures_continue_but_successes_stop(self):
        row = annotate_task(
            {
                "task_id": "retry-test",
                "branches": [
                    branch("p1", "live_valid"),
                    branch("p2", "timeout"),
                    branch("p3", "call_invalid"),
                ],
            }
        )
        by_id = {item["branch_id"]: item for item in row["branches"]}
        self.assertEqual(by_id["p1"]["continuation_target"], 0.0)
        self.assertEqual(by_id["p2"]["continuation_target"], 0.0)
        self.assertEqual(by_id["p3"]["continuation_target"], 1.0)

    def test_service_failure_is_terminal_for_the_current_task(self):
        row = annotate_task(
            {
                "task_id": "service-failure-test",
                "branches": [
                    branch("p1", "server_error"),
                    branch("p2", "live_valid"),
                ],
            }
        )
        by_id = {item["branch_id"]: item for item in row["branches"]}
        self.assertEqual(
            row["task_outcome"]["status"], "unsolvable_current_plan"
        )
        self.assertEqual(
            by_id["p1"]["effective_coherence_target"], 0.0
        )
        self.assertEqual(by_id["p1"]["continuation_target"], 0.0)
        self.assertIn(
            "abandon_tool_for_this_task",
            row["failure_report_target"],
        )


if __name__ == "__main__":
    unittest.main()
