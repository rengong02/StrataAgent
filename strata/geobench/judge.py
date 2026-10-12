"""Scoring a trajectory with GeoBenchX's rubric, on a provider we choose.

GeoBenchX's `score_task_solution` cannot be used as-is for two independent
reasons:

1. **The provider is a hard-coded whitelist.** Anything outside
   Anthropic/OpenAI/Google raises `ValueError`, so a DeepSeek judge is not
   expressible.
2. **It drives the judge through `create_react_agent(..., state_modifier=...)`**,
   an argument LangGraph 1.x removed. The judge has no tools and needs no agent
   loop, so the framework glue buys nothing and costs a broken call.

What is worth keeping is the **rubric**: `RESULT_CHECKING_PROMPT` (persona,
tool catalogue, 0/1/2 taxonomy, the detailed instructions, three worked
examples) is the thing that was tuned and validated against human annotations.
This module reuses that text verbatim and replaces only the plumbing.

Since a judge that fails silently poisons every number downstream, three
guarantees are enforced here: the raw response is always persisted, a
malformed response raises instead of yielding `None`, and usage is reported as
0 when the provider omits it rather than crashing the batch.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

#: Where the judge writes its raw replies, so a disputed score can be inspected
#: later without re-running the model.
DEFAULT_LOG = Path("evals") / "judge_raw.jsonl"


class JudgeError(RuntimeError):
    """The judge's reply could not be turned into a score."""


@dataclass
class JudgeConfig:
    """Which model scores, and how to reach it."""

    provider: str            # "openai" | "anthropic" | "deepseek" | "dashscope" | "siliconflow"
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.0
    #: Generous on purpose. A reasoning model spends this budget on thinking
    #: before it emits anything, so a tight cap yields an **empty** `content`
    #: and a parse failure that looks like a model error but is a budget error.
    #: `deepseek-flash` used 2,554 reasoning tokens on the first tuning task.
    max_tokens: int = 16_384

    def describe(self) -> str:
        """A label for the results file. Never includes the key."""
        return f"{self.provider}:{self.model}"


#: OpenAI-compatible endpoints. A named provider here means "pass this base_url
#: to the OpenAI SDK", not a separate code path.
_OPENAI_COMPATIBLE: dict[str, tuple[str | None, str]] = {
    # provider: (base_url, env var holding the key)
    "openai": (None, "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "siliconflow": ("https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"),
}

_ANTHROPIC = "anthropic"


_ENV_LOADED = False


def load_env() -> None:
    """Load `.env` from the cwd or an ancestor, once.

    CoreCoder loads it when a `Config` is built, but a judge can be constructed
    before any Agent exists -- and then `build_judge_config` reads an empty
    environment and reports "DEEPSEEK_API_KEY is not set" while the key is
    sitting in `.env` two directories up. Loading here makes the judge work
    whether or not anything else has run first.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import find_dotenv, load_dotenv

        # `usecwd=True` because import order, not the package location, decides
        # where a run starts from.
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except Exception as e:  # noqa: BLE001 - a missing dotenv must not break the judge
        # A missing `dotenv` or an unreadable file is not fatal: the keys may
        # have been exported into the process environment instead.
        log.debug("could not load .env (%s); relying on the process environment", e)


def available_judges() -> list[str]:
    """Providers that have a key in the environment right now."""
    load_env()
    found = []
    for provider, (_, env) in _OPENAI_COMPATIBLE.items():
        if os.getenv(env):
            found.append(provider)
    if os.getenv("ANTHROPIC_API_KEY"):
        found.append(_ANTHROPIC)
    return found


def build_judge_config(provider: str, model: str) -> JudgeConfig:
    """Resolve one provider and model into a config, or say what is missing.

    **`model` is required, with no default on purpose.** Model names are
    account- and time-specific -- a name that appears in a pricing table or an
    old test is not evidence that the account can reach it -- so guessing one
    would turn a clear failure into a confusing 404.

    The key is read here and passed explicitly rather than left to the SDK's
    global lookup, because `openai.api_key` is process-wide state and a stray
    `OPENAI_API_KEY` would otherwise silently win.
    """
    provider = provider.lower()
    load_env()
    if provider == _ANTHROPIC:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise JudgeError("ANTHROPIC_API_KEY is not set")
        return JudgeConfig(_ANTHROPIC, model, key)

    if provider not in _OPENAI_COMPATIBLE:
        known = ", ".join([*_OPENAI_COMPATIBLE, _ANTHROPIC])
        raise JudgeError(f"unknown judge provider {provider!r}; known: {known}")

    base_url, env = _OPENAI_COMPATIBLE[provider]
    key = os.getenv(env)
    if not key:
        raise JudgeError(f"{env} is not set, so {provider} cannot be a judge")
    return JudgeConfig(provider, model, key, base_url=base_url)


class Completer(Protocol):
    """Whatever actually sends the prompt. Swappable so tests need no network."""

    def __call__(self, system: str, user: str) -> tuple[str, dict[str, int]]: ...


def _http_completer(cfg: JudgeConfig) -> Completer:
    def complete(system: str, user: str) -> tuple[str, dict[str, int]]:
        if cfg.provider == _ANTHROPIC:
            import anthropic

            client = anthropic.Anthropic(api_key=cfg.api_key)
            resp = client.messages.create(
                model=cfg.model,
                system=system,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            usage = _usage_dict(resp.usage)
            usage["finish_reason"] = getattr(resp, "stop_reason", "?")
            usage["reasoning_chars"] = 0  # extended thinking is not requested here
            return text, usage

        import openai

        client = openai.OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        resp = client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage_obj = getattr(resp, "usage", None)
        usage = _usage_dict(usage_obj)
        message = resp.choices[0].message
        text = message.content or ""
        # Keep the reasoning trace's size as a diagnostic rather than raising
        # here: a raise inside the completer never reaches `Judge.score`, so the
        # failure would be the one kind that is not logged.
        reasoning = getattr(message, "reasoning_content", None) or ""
        usage["finish_reason"] = getattr(resp.choices[0], "finish_reason", "?")
        usage["reasoning_chars"] = len(reasoning)
        return text, usage

    return complete


def _usage_dict(usage_obj: Any) -> dict[str, int]:
    """Flatten a provider usage object, keeping cache counters.

    Keeping only prompt/completion totals loses the prompt-cache split, and the
    cache is where nearly all the input cost sits -- a report that quotes
    `prompt_cache_hit_tokens` has to be able to point at the record it came from.
    """
    if usage_obj is None:
        return {"input_tokens": 0, "output_tokens": 0}
    out: dict[str, int] = {}
    if hasattr(usage_obj, "model_dump"):
        for k, v in usage_obj.model_dump().items():
            if isinstance(v, int):
                out[k] = v
    for key, attr in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
        value = getattr(usage_obj, attr, 0)
        if isinstance(value, int):
            out[key] = value
    out.setdefault("input_tokens", 0)
    out.setdefault("output_tokens", 0)
    return out


# --- the rubric, reused verbatim from GeoBenchX ------------------------------


#: Where the fixed instructions end and the per-task payload begins.
_PAYLOAD_MARKER = "Now analyse the new candidate solution for the task"


def _load_template() -> str:
    from strata.geobench.tools import load_geobenchx_tools_module

    load_geobenchx_tools_module()  # puts the checkout on sys.path
    from geobenchx.evaluation import RESULT_CHECKING_PROMPT

    return RESULT_CHECKING_PROMPT


def split_template(template: str) -> tuple[str, str]:
    """Split the judge template into (rubric, payload template).

    Upstream sends the whole thing twice: `state_modifier=RESULT_CHECKING_PROMPT`
    puts the raw template in the system slot, and the formatted text goes in the
    user slot -- so the taxonomy, the instructions and the three worked examples
    are transmitted twice, at roughly 34k characters each.

    That reads like an oversight rather than a decision, but "probably a typo" is
    not the same as "safe to change": splitting it means our scores come from a
    different prompt than the cached panel scores did. So the split is explicit
    here, and a judge built from it is *our* pipeline -- the 47/50 figure stays a
    historical reference, not a same-pipeline baseline.
    """
    at = template.index(_PAYLOAD_MARKER)
    return template[:at].rstrip(), template[at:]


def rubric_system() -> str:
    """The fixed half: persona, tool catalogue, taxonomy, instructions, examples."""
    return split_template(_load_template())[0]


def build_prompt(task: Any) -> str:
    """Render the per-task half of the judge prompt.

    Holds the task text, the reference solutions and the candidate trajectory --
    that is, everything that changes from one task to the next.
    """
    from geobenchx.utils import get_solution_code

    if task.generated_solution is None:
        raise JudgeError("task has no generated_solution; nothing to score")

    reference = ""
    if task.reference_solution_description is not None:
        reference = (
            "Here are general comments about the reference solutions: "
            f"{task.reference_solution_description}"
        )
    for solution in task.reference_solutions:
        reference += (
            f"<REFERENCE SOLUTION>\n{get_solution_code(solution, add_num=False)}\n"
            "</REFERENCE SOLUTION>"
        )

    return split_template(_load_template())[1].format(
        task_text=task.task_text,
        reference_solutions=reference,
        candidate_solution=get_solution_code(task.generated_solution, add_num=False),
    )


def _score_values(value: int):
    """Wrap a score in GeoBenchX's `ScoreValues` enum.

    Its statistics code reads `task.match_score_LLM.value`; assigning a plain
    `int` succeeds here and fails there.
    """
    from geobenchx.constants import ScoreValues

    return ScoreValues(value)


_REASONING_RE = re.compile(r"<MATCH REASONING>(.*?)</MATCH REASONING>", re.DOTALL)
_SCORE_RE = re.compile(r"<MATCH SCORE>(.*?)</MATCH SCORE>", re.DOTALL)


def parse_verdict(text: str) -> tuple[str, int]:
    """Pull the reasoning and the 0/1/2 score out of a judge reply.

    Raises rather than returning a sentinel: a `None` score silently skipped by
    the statistics code would shrink the denominator without anyone noticing.
    """
    score_match = _SCORE_RE.search(text)
    if not score_match:
        raise JudgeError(f"judge reply has no <MATCH SCORE>: {text[-400:]!r}")
    raw = score_match.group(1).strip()
    try:
        score = int(raw)
    except ValueError as e:
        raise JudgeError(f"judge score is not an integer: {raw!r}") from e
    if score not in (0, 1, 2):
        raise JudgeError(f"judge score outside the 0/1/2 taxonomy: {score}")

    reasoning_match = _REASONING_RE.search(text)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    return reasoning, score


# --- scoring ------------------------------------------------------------------


@dataclass
class Judge:
    """Scores trajectories, and keeps a record of what it was told."""

    config: JudgeConfig
    complete: Completer | None = None
    log_path: Path | None = DEFAULT_LOG
    #: Filled in as tasks are scored; handy for a cost line in the report.
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})

    def __post_init__(self):
        if self.complete is None:
            self.complete = _http_completer(self.config)

    def score(self, task: Any) -> tuple[str, int]:
        system = rubric_system()
        user = build_prompt(task)
        raw, usage = self.complete(system, user)  # type: ignore[misc]
        self.usage["input_tokens"] += usage.get("input_tokens", 0)
        self.usage["output_tokens"] += usage.get("output_tokens", 0)

        try:
            if not raw.strip():
                # Reasoning models put the answer in `content` and the working in
                # `reasoning_content`; empty content with a long reasoning trace
                # means the token budget ran out mid-thought. That reads like a
                # model failure and is a budget failure, so name it.
                raise JudgeError(
                    "judge returned empty content "
                    f"(finish_reason={usage.get('finish_reason')}, "
                    f"reasoning_chars={usage.get('reasoning_chars')}, "
                    f"max_tokens={self.config.max_tokens}); raise max_tokens if "
                    "the model spent the budget thinking"
                )
            reasoning, value = parse_verdict(raw)
        except JudgeError:
            # Log the failure too. A reply that could not be parsed is exactly
            # the one somebody will want to read later, and the first version of
            # this skipped the log on that path -- so the 50-task calibration
            # reported "1 parse failure" with no record of what came back.
            self._log(task, raw, None, usage)
            raise

        task.match_reasoning_LLM = reasoning
        # GeoBenchX's own statistics read `task.match_score_LLM.value`, so a bare
        # int here raises AttributeError several steps later, far from the cause.
        task.match_score_LLM = _score_values(value)
        self._log(task, raw, value, usage)
        return reasoning, value

    def _log(self, task: Any, raw: str, value: int | None, usage: dict[str, int]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "task_id": getattr(task, "task_ID", ""),
            "judge": self.config.describe(),
            "score": value,  # None means the reply could not be parsed
            "usage": usage,
            "raw": raw,  # the whole reply, so a disputed score stays inspectable
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def majority_score(scores: list[int]) -> int:
    """The score most runs agreed on, ties going to the *lower* one.

    Repeating a judgement is only worth anything if the repeats are combined by
    a stated rule. This one is deliberately conservative: a judge that cannot
    settle between "partial" and "full" has not decided the task is a full
    match, and reading the tie upward would quietly inflate the number --
    in the direction that flatters the thing being measured.
    """
    if not scores:
        raise ValueError("majority_score needs at least one score")
    counts = Counter(scores)
    top = max(counts.values())
    return min(score for score, count in counts.items() if count == top)


def calibration_report(judge_scores: list[int], human_scores: list[int]) -> dict:
    """Agreement between a judge and the human annotations.

    GeoBenchX ships 50 human-scored tasks precisely so a substitute judge can be
    checked before its numbers are trusted. Exact agreement is the headline;
    adjacency matters too, because 0-vs-1 is a milder disagreement than 0-vs-2.
    """
    if len(judge_scores) != len(human_scores):
        raise JudgeError("judge and human score lists differ in length")
    n = len(judge_scores)
    if n == 0:
        raise JudgeError("no scored tasks to compare")

    exact = sum(1 for j, h in zip(judge_scores, human_scores) if j == h)
    adjacent = sum(1 for j, h in zip(judge_scores, human_scores) if abs(j - h) <= 1)
    matrix = {
        f"{h}->{j}": sum(1 for a, b in zip(judge_scores, human_scores) if b == h and a == j)
        for h in (0, 1, 2) for j in (0, 1, 2)
    }
    return {
        "n": n,
        "exact_agreement": exact / n,
        "within_one": adjacent / n,
        "confusion": matrix,
    }
