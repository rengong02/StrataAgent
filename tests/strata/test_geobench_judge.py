"""The judge: rubric reuse, provider choice, parsing, and calibration.

The reason this is worth testing rather than trusting: every number the project
will ever report passes through here. A judge that silently returns `None`
shrinks the denominator; one that mis-parses a score shifts every result. So the
failure modes are made loud on purpose.

The 50-task tuning set is the anchor. GeoBenchX ships it with human scores
*and* the scores its own GPT-4.1 panel produced, and those two agree on 47/50.
That gives a fixed, offline reference for the calibration maths -- no API call
needed to know whether the harness measures agreement correctly.
"""

from __future__ import annotations

import json

import pytest

from strata.geobench.judge import (
    Judge,
    JudgeConfig,
    JudgeError,
    build_judge_config,
    calibration_report,
    parse_verdict,
)
from strata.geobench.tools import geobenchx_root, load_geobenchx_tools_module

load_geobenchx_tools_module()  # puts the GeoBenchX checkout on sys.path
ROOT = geobenchx_root()
TUNING = ROOT / "benchmark_set" / "evaluator_tuning_set.json"

requires_tuning = pytest.mark.skipif(
    not TUNING.exists(), reason="GeoBenchX tuning set not present"
)


# --- parsing, where a quiet failure would poison everything downstream -------


def test_a_well_formed_verdict_parses():
    text = "<MATCH REASONING>same data, same tools</MATCH REASONING>\n<MATCH SCORE>2</MATCH SCORE>"
    reasoning, score = parse_verdict(text)
    assert score == 2
    assert "same data" in reasoning


@pytest.mark.parametrize("bad", [
    "no tags at all",
    "<MATCH REASONING>only reasoning</MATCH REASONING>",
    "<MATCH SCORE>two</MATCH SCORE>",
])
def test_a_malformed_verdict_raises_rather_than_returning_none(bad):
    with pytest.raises(JudgeError):
        parse_verdict(bad)


def test_a_score_outside_the_taxonomy_is_rejected():
    """The rubric defines 0/1/2; a 3 means the judge went off-script."""
    with pytest.raises(JudgeError, match="taxonomy"):
        parse_verdict("<MATCH SCORE>3</MATCH SCORE>")


def test_reasoning_is_optional_but_the_score_is_not():
    """A terse judge that only emits the score is usable; one without is not."""
    _, score = parse_verdict("<MATCH SCORE>1</MATCH SCORE>")
    assert score == 1


# --- provider selection -------------------------------------------------------


def test_each_known_provider_resolves(monkeypatch):
    for provider, env in [
        ("openai", "OPENAI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("dashscope", "DASHSCOPE_API_KEY"),
        ("siliconflow", "SILICONFLOW_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
    ]:
        monkeypatch.setenv(env, "test-key")
        cfg = build_judge_config(provider, "some-model-id")
        assert cfg.api_key == "test-key"
        assert cfg.provider == provider


def test_a_missing_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(JudgeError, match="DEEPSEEK_API_KEY"):
        build_judge_config("deepseek", "m")


def test_an_unknown_provider_lists_the_known_ones(monkeypatch):
    with pytest.raises(JudgeError, match="known"):
        build_judge_config("some-other-llm", "m")


def test_the_description_never_leaks_the_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-value")
    cfg = build_judge_config("deepseek", "some-model-id")
    assert "secret" not in cfg.describe()


def test_deepseek_gets_its_own_base_url(monkeypatch):
    """The whole point of a provider builder: an explicit endpoint, not a global."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    cfg = build_judge_config("deepseek", "some-model-id")
    assert cfg.base_url and "deepseek" in cfg.base_url
    # and it must not be silently inherited from OPENAI_BASE_URL
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid")
    assert "example.invalid" not in (build_judge_config("deepseek", "m").base_url or "")


# --- scoring, with the network replaced by a scripted reply -------------------


class _Scripted:
    def __init__(self, reply: str, usage=None):
        self.reply = reply
        self.usage = usage or {"input_tokens": 10, "output_tokens": 5}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str):
        self.calls.append((system, user))
        return self.reply, self.usage


def _config():
    return JudgeConfig("deepseek", "deepseek-chat", "test-key", base_url="http://x")


@requires_tuning
def test_scoring_a_real_task_fills_in_the_fields_and_logs(tmp_path):
    """Uses the tuning set, not the reference set: only the former ships a
    `generated_solution`, and a task without one has nothing to score."""
    from geobenchx.dataclasses import Task

    task = Task(**json.loads(TUNING.read_text(encoding="utf-8"))["tasks"][0])
    completer = _Scripted("<MATCH REASONING>looks right</MATCH REASONING><MATCH SCORE>2</MATCH SCORE>")
    judge = Judge(_config(), complete=completer, log_path=tmp_path / "raw.jsonl")

    reasoning, score = judge.score(task)

    assert (score, reasoning) == (2, "looks right")
    assert task.match_score_LLM == 2
    assert judge.usage == {"input_tokens": 10, "output_tokens": 5}

    logged = json.loads((tmp_path / "raw.jsonl").read_text(encoding="utf-8").strip())
    assert logged["score"] == 2
    assert "<MATCH SCORE>2</MATCH SCORE>" in logged["raw"], "the raw reply must be kept"


@requires_tuning
def test_the_prompt_carries_the_rubric_and_both_solutions():
    from geobenchx.dataclasses import Task

    task = Task(**json.loads(TUNING.read_text(encoding="utf-8"))["tasks"][0])

    completer = _Scripted("<MATCH SCORE>2</MATCH SCORE>")
    Judge(_config(), complete=completer, log_path=None).score(task)

    system, user = completer.calls[0]
    assert "<MATCH SCORE>" in system, "the rubric (with its output tags) is the system prompt"
    assert task.task_text in user
    assert "<REFERENCE SOLUTION>" in user
    assert "candidate" in user.lower()


def test_usage_defaults_to_zero_when_the_provider_omits_it():
    """A missing counter must not abort a 50-task calibration run."""
    judge = Judge(_config(), complete=_Scripted("<MATCH SCORE>1</MATCH SCORE>", usage={}), log_path=None)
    assert judge.usage == {"input_tokens": 0, "output_tokens": 0}


def test_an_unparseable_reply_raises_out_of_score():
    judge = Judge(_config(), complete=_Scripted("I think it matches?"), log_path=None)
    with pytest.raises(JudgeError):
        judge.score(type("T", (), {"task_text": "x", "generated_solution": None})())


# --- calibration, checked against GeoBenchX's own stored panel scores ---------


@requires_tuning
def test_the_stored_panel_scores_reproduce_their_known_agreement():
    """A fixed reference for the maths: GPT-4.1 agrees with humans on 47/50.

    Running this needs no API key, so it also proves the harness measures what
    the paper says it measures before we point a different model at it.
    """
    tasks = json.loads(TUNING.read_text(encoding="utf-8"))["tasks"]
    judge_scores = [t["match_score_LLM"] for t in tasks]
    human_scores = [t["match_score_Human"] for t in tasks]

    report = calibration_report(judge_scores, human_scores)

    assert report["n"] == 50
    assert report["exact_agreement"] == pytest.approx(47 / 50)
    assert report["within_one"] == pytest.approx(49 / 50)
    assert sum(report["confusion"].values()) == 50


def test_calibration_rejects_mismatched_inputs():
    with pytest.raises(JudgeError):
        calibration_report([1, 2], [1])
    with pytest.raises(JudgeError):
        calibration_report([], [])


def test_calibration_confusion_matrix_is_readable():
    """Row = human, column = judge, so a systematic bias is visible."""
    report = calibration_report([2, 2, 0], [1, 2, 0])
    assert report["exact_agreement"] == pytest.approx(2 / 3)
    assert report["confusion"]["1->2"] == 1  # human said 1, judge said 2


# --- judge response and logging regressions ----------------------------------


def test_env_file_alone_is_enough(tmp_path, monkeypatch):
    """The judge must load `.env` itself, not rely on something else having run.

    Constructing a judge before any Agent means `build_judge_config` is the first
    thing to read the environment; without this it reports a missing key while
    the key sits in `.env`.
    """
    from strata.geobench import judge as J

    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=from-dotenv-file\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(J, "_ENV_LOADED", False)

    cfg = build_judge_config("deepseek", "some-model-id")

    assert cfg.api_key == "from-dotenv-file"


def test_a_process_variable_still_wins_over_the_file(tmp_path, monkeypatch):
    """`override=False`: an explicit export beats a checked-in default."""
    from strata.geobench import judge as J

    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-process")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(J, "_ENV_LOADED", False)

    assert build_judge_config("deepseek", "m").api_key == "from-process"


@requires_tuning
def test_the_score_is_stored_as_a_score_values_enum(tmp_path):
    """GeoBenchX's statistics read `match_score_LLM.value`.

    A bare int assigns fine and then blows up several steps later, far from here.
    """
    from geobenchx.constants import ScoreValues
    from geobenchx.dataclasses import Task

    task = Task(**json.loads(TUNING.read_text(encoding="utf-8"))["tasks"][0])
    judge = Judge(_config(), complete=_Scripted("<MATCH SCORE>2</MATCH SCORE>"), log_path=None)

    judge.score(task)

    assert isinstance(task.match_score_LLM, ScoreValues)
    assert task.match_score_LLM.value == 2  # what the stats code actually calls


def test_the_rubric_is_not_sent_twice():
    """Upstream repeats the whole template; we split it.

    Measured upstream: ~33.7k chars of system and ~34.8k chars of user, both
    containing the taxonomy and examples. Repeating it doubles the input cost and
    changes nothing about the task.
    """
    from strata.geobench.judge import rubric_system, split_template

    template = "RUBRIC PART\n\n" + "Now analyse the new candidate solution for the task\nPAYLOAD {task_text}"

    rubric, payload = split_template(template)

    assert rubric == "RUBRIC PART"
    assert rubric_system and payload.startswith("Now analyse")


@requires_tuning
def test_system_holds_the_rubric_and_user_holds_only_the_task():
    from geobenchx.dataclasses import Task

    task = Task(**json.loads(TUNING.read_text(encoding="utf-8"))["tasks"][0])
    completer = _Scripted("<MATCH SCORE>1</MATCH SCORE>")
    Judge(_config(), complete=completer, log_path=None).score(task)

    system, user = completer.calls[0]
    assert "MATCH SCORE" in system, "the rubric belongs in the system slot"
    assert task.task_text in user
    assert "EVALUATION TAXONOMY" not in user, "the taxonomy was sent a second time"
    assert "<CANDIDATE SOLUTION>" in user
    assert len(user) < len(system), "user should be the small, per-task half"


# --- the production completer, checked without a network ----------------------


def test_the_openai_completer_passes_key_url_and_model(monkeypatch):
    """A wire contract: the fake and the real path agree on what gets sent.

    Without this, the only evidence the production completer is wired correctly
    would be one live smoke run -- which is not repeatable in CI.
    """
    import sys
    import types

    seen: dict = {}

    class _Msg:
        content = "<MATCH SCORE>2</MATCH SCORE>"

    class _Choice:
        message = _Msg()

    class _Usage:
        prompt_tokens = 7
        completion_tokens = 3

    class _Resp:
        def __init__(self):
            self.choices = [_Choice()]
            self.usage = _Usage()

    class _Completions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return _Resp()

    class _Client:
        def __init__(self, **kwargs):
            seen["client"] = kwargs
            self.chat = types.SimpleNamespace(completions=_Completions())

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    from strata.geobench.judge import _http_completer

    cfg = JudgeConfig("deepseek", "model-x", "key-y", base_url="https://api.example/v1",
                      temperature=0.0, max_tokens=123)
    text, usage = _http_completer(cfg)("sys", "user")

    assert seen["client"] == {"api_key": "key-y", "base_url": "https://api.example/v1"}
    assert seen["model"] == "model-x" and seen["max_tokens"] == 123
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]
    # usage now also carries the diagnostics the empty-content path needs
    assert usage["input_tokens"] == 7 and usage["output_tokens"] == 3
    assert "finish_reason" in usage and "reasoning_chars" in usage
    assert "<MATCH SCORE>" in text


# --- a reasoning model's empty content, and failures that were not logged -----


def test_empty_content_from_a_reasoning_model_says_why(monkeypatch):
    """`deepseek-flash` spends `max_tokens` thinking before it writes anything.

    With a tight cap the reply comes back as an empty string -- which looks like
    a model failure and is actually a budget failure. Say so.
    """
    import sys
    import types

    class _Msg:
        content = ""
        reasoning_content = "x" * 11497

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Resp:
        def __init__(self):
            self.choices = [_Choice()]
            self.usage = None

    class _Completions:
        def create(self, **kwargs):
            return _Resp()

    class _Client:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=_Completions())

    fake = types.ModuleType("openai")
    fake.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    from strata.geobench.judge import _http_completer

    cfg = JudgeConfig("deepseek", "m", "k", base_url="http://x", max_tokens=4096)
    text, usage = _http_completer(cfg)("s", "u")

    # The completer reports rather than raises: a raise here would happen before
    # `Judge.score` can log it, leaving the one failure nobody can inspect.
    assert text == ""
    assert usage["reasoning_chars"] == 11497
    assert usage["finish_reason"] == "stop"


@requires_tuning
def test_an_unparseable_reply_is_still_logged(tmp_path):
    """The reply you most want to read later is the one that failed to parse."""
    from geobenchx.dataclasses import Task

    task = Task(**json.loads(TUNING.read_text(encoding="utf-8"))["tasks"][0])
    log = tmp_path / "raw.jsonl"
    judge = Judge(_config(), complete=_Scripted("I cannot score this."), log_path=log)

    with pytest.raises(JudgeError):
        judge.score(task)

    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1, "the failed reply was not recorded"
    assert rows[0]["score"] is None
    assert rows[0]["raw"] == "I cannot score this."


def test_the_default_token_budget_leaves_room_for_thinking():
    """A 4096 cap is smaller than the reasoning trace a judge model produces."""
    assert JudgeConfig("deepseek", "m", "k").max_tokens >= 8192


@requires_tuning
def test_an_empty_reply_is_logged_before_it_raises(tmp_path):
    """The empty-content path must reach the log too.

    Diagnostics raised from
    inside the completer, so this class of failure never met `_log()`.
    """
    from geobenchx.dataclasses import Task

    task = Task(**json.loads(TUNING.read_text(encoding="utf-8"))["tasks"][0])

    def empty(system, user):
        return "", {"input_tokens": 5, "output_tokens": 0,
                    "finish_reason": "stop", "reasoning_chars": 9999}

    log = tmp_path / "raw.jsonl"
    judge = Judge(_config(), complete=empty, log_path=log)

    with pytest.raises(JudgeError, match="max_tokens"):
        judge.score(task)

    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["score"] is None and rows[0]["raw"] == ""
    assert rows[0]["usage"]["reasoning_chars"] == 9999


def test_full_usage_is_kept_not_just_the_totals():
    """The report quotes prompt-cache numbers; the log has to back them up."""
    from strata.geobench.judge import _usage_dict

    class _U:
        prompt_tokens = 8912
        completion_tokens = 2891
        prompt_cache_hit_tokens = 8704
        prompt_cache_miss_tokens = 208

        def model_dump(self):
            return {"prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "prompt_cache_hit_tokens": self.prompt_cache_hit_tokens,
                    "prompt_cache_miss_tokens": self.prompt_cache_miss_tokens}

    u = _usage_dict(_U())
    assert u["input_tokens"] == 8912 and u["output_tokens"] == 2891
    assert u["prompt_cache_hit_tokens"] == 8704, "the cache split was dropped"


# --- combining repeated judgements -------------------------------------------
#
# Repeating a judgement is only worth something if the repeats are combined by a
# stated rule, and the rule has to be one that cannot flatter the result.


def test_the_majority_wins_when_there_is_one():
    from strata.geobench.judge import majority_score

    assert majority_score([2, 2, 1]) == 2
    assert majority_score([0, 1, 1]) == 1


def test_a_tie_goes_to_the_lower_score():
    """A judge that cannot settle has not decided the task is a full match.

    Reading the tie upward would inflate the number in the direction that
    flatters whatever is being measured.
    """
    from strata.geobench.judge import majority_score

    assert majority_score([1, 2]) == 1
    assert majority_score([0, 2]) == 0
    assert majority_score([2, 2, 0, 0]) == 0


def test_one_judgement_needs_no_combining():
    from strata.geobench.judge import majority_score

    assert majority_score([2]) == 2
    assert majority_score([0]) == 0


def test_no_scores_is_an_error_rather_than_a_default():
    """A silent 0 would read as "the model scored nothing"."""
    import pytest

    from strata.geobench.judge import majority_score

    with pytest.raises(ValueError, match="at least one"):
        majority_score([])
