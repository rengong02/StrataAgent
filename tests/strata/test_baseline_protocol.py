"""Protocol-only checks for the local baseline driver.

These tests do not load a checkpoint or the GIS stack. They pin the transcript
shape that a model run must use before any expensive baseline is repeated.
"""

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "run_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_baseline", SCRIPT)
assert SPEC and SPEC.loader
run_baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_baseline)


def test_tool_result_is_paired_with_a_structured_assistant_call():
    messages = [{"role": "system", "content": "tools"},
                {"role": "user", "content": "task"}]
    updated = run_baseline.append_tool_turn(
        messages, "call_1",
        {"name": "load_data", "arguments": {"dataset": "GDP"}},
        "loaded",
    )

    assert updated[:2] == messages
    assert updated[2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "load_data", "arguments": {"dataset": "GDP"}},
        }],
    }
    assert updated[3] == {
        "role": "tool", "tool_call_id": "call_1", "content": "loaded"
    }


def test_run_one_preserves_raw_tool_observation_separately_from_model_termination(
        monkeypatch):
    class FakeTensor:
        shape = (1, 1)

        def __getitem__(self, _item):
            return self

    class FakeNoGrad:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = FakeNoGrad
    fake_torch.tensor = lambda _values: FakeTensor()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeInputs(dict):
        def __init__(self):
            super().__init__(input_ids=FakeTensor())

        def to(self, _device):
            return self

    class FakeTokenizer:
        eos_token_id = 0

        def __init__(self):
            self.decode_count = 0

        def apply_chat_template(self, messages, tools=None, **kwargs):
            return "rendered"

        def __call__(self, prompt, return_tensors=None):
            return FakeInputs()

        def decode(self, output, skip_special_tokens=True):
            self.decode_count += 1
            if self.decode_count == 1:
                return ('<tool_call>{"name":"merge_dataframes",'
                        '"arguments":{"statkey":"bad"}}</tool_call>')
            return ('{"status":"error","schema_version":2,'
                    '"error_code":"merge","error_message":"gave up"}')

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            return FakeTensor()

    class FakeServer:
        def call(self, tool, arguments):
            assert tool == "merge_dataframes"
            assert arguments == {"statkey": "bad"}
            return {"observation": "Error: missing geokey", "failed": True}

    result = run_baseline.run_one(
        FakeModel(), FakeTokenizer(),
        {"messages": [{"role": "system", "content": "system"},
                      {"role": "user", "content": "task"}],
         "tools": [{"type": "function", "function": {"name": "merge_dataframes"}}]},
        FakeServer(), max_turns=2, max_new_tokens=32,
    )

    assert result["attempt"]["calls"] == [{
        "id": "call_1",
        "name": "merge_dataframes",
        "arguments": {"statkey": "bad"},
    }]
    assert result["tool_observations"] == [{
        "call_id": "call_1",
        "tool": "merge_dataframes",
        "arguments": {"statkey": "bad"},
        "observation": "Error: missing geokey",
        "failed": True,
    }]
    assert result["attempt"]["termination"]["error_message"] == "gave up"


def test_sample_digest_changes_when_the_sample_protocol_changes():
    first = [{"sample_id": "s", "schema_version": 1}]
    second = [{"sample_id": "s", "schema_version": 2}]
    assert run_baseline.samples_digest(first) != run_baseline.samples_digest(second)


def test_sample_digest_ignores_wall_clock_step_timings():
    first = [{"sample_id": "s", "steps": [{"tool": "x", "seconds": 0.0002}]}]
    second = [{"sample_id": "s", "steps": [{"tool": "x", "seconds": 0.0007}]}]
    assert run_baseline.samples_digest(first) == run_baseline.samples_digest(second)


def test_sample_digest_keeps_step_protocol_changes():
    first = [{"sample_id": "s", "steps": [{"tool": "x", "seconds": 0.0002}]}]
    second = [{"sample_id": "s", "steps": [{"tool": "y", "seconds": 0.0002}]}]
    assert run_baseline.samples_digest(first) != run_baseline.samples_digest(second)


def test_auto_device_prefers_cuda_then_mps_then_cpu():
    cuda = SimpleNamespace(is_available=lambda: True)
    mps = SimpleNamespace(is_available=lambda: True)
    torch_module = SimpleNamespace(cuda=cuda, backends=SimpleNamespace(mps=mps))

    assert run_baseline.select_device(torch_module, "auto") == "cuda"


def test_explicit_unavailable_device_is_rejected():
    cuda = SimpleNamespace(is_available=lambda: False)
    mps = SimpleNamespace(is_available=lambda: False)
    torch_module = SimpleNamespace(cuda=cuda, backends=SimpleNamespace(mps=mps))

    try:
        run_baseline.select_device(torch_module, "cuda")
    except RuntimeError as exc:
        assert "cuda" in str(exc)
    else:
        raise AssertionError("unavailable explicit device should be rejected")


def test_auto_attention_uses_eager_on_mps_and_sdpa_elsewhere():
    assert run_baseline.select_attention_implementation("mps", "auto") == "eager"
    assert run_baseline.select_attention_implementation("cuda", "auto") == "sdpa"
    assert run_baseline.select_attention_implementation("cpu", "eager") == "eager"


def test_model_loader_uses_transformers_dtype_keyword():
    calls = {}

    class Loader:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.update(kwargs)
            return "model"

    assert run_baseline.load_checkpoint(Loader, "checkpoint", "sdpa") == "model"
    assert calls == {"torch_dtype": "auto", "attn_implementation": "sdpa"}


def test_catalog_tools_are_loaded_only_when_requested():
    tool_server = importlib.import_module("scripts.tool_server")
    sentinel = object()
    fake_build = types.ModuleType("strata.train.build")
    fake_build.fresh_tools = lambda: sentinel
    previous = sys.modules.get("strata.train.build")
    sys.modules["strata.train.build"] = fake_build
    try:
        assert tool_server.load_catalog_tools() is sentinel
    finally:
        if previous is None:
            sys.modules.pop("strata.train.build", None)
        else:
            sys.modules["strata.train.build"] = previous


def test_synthetic_samples_do_not_require_catalog_rate_probe():
    synthetic = {"source_provenance": {"source_kind": "synthetic_generator"}}
    catalog = {"source_provenance": {"source_kind": "geobenchx_catalog"}}
    assert run_baseline.requires_catalog_rates([synthetic]) is False
    assert run_baseline.requires_catalog_rates([synthetic, catalog]) is True


def test_probe_evaluation_is_not_described_as_an_untrained_checkpoint():
    assert "untrained" in run_baseline.evaluation_description("base")["what"]
    probe = run_baseline.evaluation_description("sft-probe")
    assert "untrained" not in probe["what"]
    assert "probe" in probe["what"]
    assert "exploratory" in probe["not"]


@pytest.mark.parametrize("kind_option", ["--checkpoint-kind", "--evaluation-kind"])
def test_evaluation_cli_aliases_preserve_checkpoint_metadata(monkeypatch, tmp_path, kind_option):
    samples = tmp_path / "empty_samples.jsonl"
    samples.write_text("", encoding="utf-8")
    output = tmp_path / "report.json"
    model = SimpleNamespace(to=lambda device: None, eval=lambda: None)
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object()
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda name: object())
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(run_baseline, "select_device", lambda module, device: "cpu")
    monkeypatch.setattr(run_baseline, "load_checkpoint", lambda loader, name, attention: model)
    monkeypatch.setattr(
        run_baseline, "ToolServer",
        lambda python_path: SimpleNamespace(fill_rates=dict, close=lambda: None),
    )
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "--samples", str(samples), "--out", str(output),
        kind_option, "sft-probe",
    ])

    assert run_baseline.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["checkpoint_kind"] == report["evaluation_kind"] == "sft-probe"
    assert "untrained" not in report["what"]
    assert report["summary"]["n"] == 0


def test_evaluation_description_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown evaluation kind"):
        run_baseline.evaluation_description("sft-prboe")


@pytest.mark.parametrize("kind_option", ["--checkpoint-kind", "--evaluation-kind"])
def test_evaluation_cli_rejects_invalid_kind(monkeypatch, tmp_path, capsys, kind_option):
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "--samples", str(tmp_path / "missing.jsonl"),
        "--out", str(tmp_path / "report.json"), kind_option, "sft-prboe",
    ])

    with pytest.raises(SystemExit) as failure:
        run_baseline.main()
    assert failure.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
