import unittest

from api_harness_policy import PromptHarnessPolicy, extract_json


class PromptHarnessPolicyTest(unittest.TestCase):
    def test_extracts_fenced_json(self):
        self.assertEqual(
            extract_json('```json\n{"actions":[]}\n```'),
            {"actions": []},
        )

    def test_normalizes_plan_and_dependencies(self):
        def generate(_requests):
            return [{
                "text": """{"actions":[
                    {"id":"a","toolName":"resolve","objective":"resolve Alice"},
                    {"id":"b","toolName":"orders","objective":"fetch orders",
                     "dependsOn":["a"],"executionWave":2}
                ]}""",
                "promptTokens": 20,
                "outputTokens": 30,
            }]

        policy = PromptHarnessPolicy(generate)
        actions = policy.plan({
            "task": "Resolve Alice then fetch orders",
            "maxBranches": 4,
            "tools": [
                {"name": "resolve", "description": "resolve customer"},
                {"name": "orders", "description": "fetch orders"},
            ],
        })["actions"]
        self.assertEqual([item["toolName"] for item in actions], [
            "resolve",
            "orders",
        ])
        self.assertEqual(actions[1]["dependsOn"], ["a"])
        self.assertFalse(actions[1]["ready"])

    def test_keeps_success_and_terminal_failure_evidence(self):
        def generate(_requests):
            return [{
                "text": """{"scores":[],"taskComplete":true,
                    "taskSolvable":false,"mainDecision":"takeover"}""",
                "promptTokens": 20,
                "outputTokens": 20,
            }]

        policy = PromptHarnessPolicy(generate)
        selected = policy.select({
            "task": "Check weather and inventory",
            "actions": [{"id": "weather"}, {"id": "inventory"}],
            "drafts": [
                {
                    "briefId": "weather",
                    "status": "ok",
                    "toolCalls": [{
                        "toolName": "weather",
                        "isError": False,
                        "resultText": "SUNNY",
                    }],
                },
                {
                    "briefId": "inventory",
                    "status": "ok",
                    "toolCalls": [{
                        "toolName": "inventory",
                        "isError": True,
                        "resultText": "503 Service Unavailable",
                    }],
                },
            ],
        })
        self.assertTrue(all(item["retain"] for item in selected["scores"]))
        self.assertTrue(selected["taskComplete"])
        self.assertEqual(
            selected["failureReports"][0]["failureType"],
            "server_or_network",
        )


if __name__ == "__main__":
    unittest.main()
