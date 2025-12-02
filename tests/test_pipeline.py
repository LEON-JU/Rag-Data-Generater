from __future__ import annotations

from rag_data_generator.evaluation.answer import AnswerEvaluator, EvaluationItem
from rag_data_generator.pipeline.interruption import InterruptionOrchestrator
from rag_data_generator.tooling.registry import ToolRegistry
from rag_data_generator.tooling.tools import ToolSpec


class DummyLLMClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("No responses left in dummy client")
        return self._responses.pop(0)


class RecordingTool:
    def __init__(self):
        self.spec = ToolSpec(
            name_for_human="Recorder",
            name_for_model="Wiki_RAG",
            description_for_model="records invocations",
            parameters=[],
        )
        self.invocations = []

    def invoke(self, payload):
        self.invocations.append(payload)
        return {"echo": payload.get("input")}


class StaticEvalClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def complete(self, prompt):
        if not self.outputs:
            raise AssertionError("No outputs configured")
        return self.outputs.pop(0)


def test_interruption_controller_handles_tool_invocation():
    dummy_llm = DummyLLMClient(
        [
            "<think>Need facts</think><search>[Wiki_RAG]: nemo florida</search>",
            "<answer>nemo was spotted in tarpon springs.</answer>",
        ]
    )
    tool = RecordingTool()
    registry = ToolRegistry(tools=[tool])
    orchestrator = InterruptionOrchestrator(llm_client=dummy_llm, tool_registry=registry)
    initial_messages = [{"role": "user", "content": "Where was the clownfish seen?"}]
    result = orchestrator.run(initial_messages)

    assert tool.invocations, "tool should be triggered when <search> appears"
    assert result["history"][-1]["content"].endswith("</answer>"), "final entry should be assistant reply"
    assert any("<observation>" in entry["content"] for entry in result["history"]), "observation missing"


def test_answer_evaluator_tracks_accuracy():
    evaluator = AnswerEvaluator(client=StaticEvalClient(["Yes", "No"]))
    items = [
        EvaluationItem(question="Q1", expected="A", predicted="A"),
        EvaluationItem(question="Q2", expected="B", predicted="C"),
    ]
    correct, total, accuracy, records = evaluator.evaluate(items)

    assert correct == 1 and total == 2
    assert accuracy == 0.5
    assert records[0]["eval"] == "true" and records[1]["eval"] == "false"
