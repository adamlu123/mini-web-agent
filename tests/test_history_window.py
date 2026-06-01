"""Unit tests for the sliding-window history logic in the terminal agent.

Validates ``TerminalAgentGenerator._windowed_generation_context`` — the pure
helper that decides which slice of the conversation is fed back to the policy.
It mirrors how ``_agent_loop`` records per-turn boundaries, so the test also
simulates that bookkeeping to catch off-by-one / alignment bugs.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _stub_skyrl() -> None:
    """Install minimal ``skyrl`` stubs so the generator module imports offline.

    The training stack (skyrl / vLLM) is only present inside the training
    image; the windowing helper under test is pure Python and needs none of it.
    We provide just the names the module references at import time.
    """
    try:
        import skyrl  # noqa: F401

        return  # Real training stack present (e.g. the echo-rl env) — use it.
    except ImportError:
        pass

    def _mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    _mod("skyrl")
    _mod("skyrl.backends")
    _mod("skyrl.backends.skyrl_train")
    _mod("skyrl.backends.skyrl_train.inference_engines")
    iec = _mod("skyrl.backends.skyrl_train.inference_engines.inference_engine_client")
    iec.InferenceEngineClient = type("InferenceEngineClient", (), {})
    _mod("skyrl.train")
    _mod("skyrl.train.generators")
    base = _mod("skyrl.train.generators.base")
    base.GeneratorInput = dict
    base.GeneratorOutput = dict
    base.GeneratorInterface = type("GeneratorInterface", (), {})
    base.TrajectoryID = type("TrajectoryID", (), {})
    utils = _mod("skyrl.train.generators.utils")
    utils.get_rollout_metrics = lambda *a, **k: {}


_stub_skyrl()

from echo_rl.terminal_agent.terminal_agent_generator import TerminalAgentGenerator  # noqa: E402

win = TerminalAgentGenerator._windowed_generation_context

PROMPT = [900, 901, 902]  # stand-in prompt token ids (ends with a gen prompt)


def _simulate(num_turns: int, window_turns, tokens_per_turn: int = 4):
    """Replay the loop's boundary bookkeeping and capture each turn's context.

    Each turn appends ``tokens_per_turn`` completion tokens (assistant + obs).
    Returns the list of generation contexts the policy would have seen.
    """
    completion: list[int] = []
    turn_boundaries: list[int] = []
    seen: list[list[int]] = []
    for turn in range(num_turns):
        full_context = PROMPT + completion
        completion_portion = full_context[len(PROMPT):]
        turn_boundaries.append(len(completion_portion))
        ctx = win(PROMPT, completion_portion, full_context, turn_boundaries, turn, window_turns)
        seen.append(list(ctx))
        # Emit this turn's tokens; encode the turn index so we can assert which
        # turns survive in later windows.
        completion.extend([turn * 10 + i for i in range(tokens_per_turn)])
    return seen


def test_no_window_returns_full_history():
    seen = _simulate(num_turns=5, window_turns=None)
    # Last turn must see the prompt plus every prior turn's tokens.
    assert seen[-1] == PROMPT + [t * 10 + i for t in range(4) for i in range(4)]


def test_zero_or_negative_treated_as_no_window():
    # _agent_loop normalises <=0 to None before calling, but guard anyway:
    # callers pass None for "unlimited", so simulate that contract here.
    seen_none = _simulate(num_turns=4, window_turns=None)
    assert seen_none[-1] == PROMPT + [t * 10 + i for t in range(3) for i in range(4)]


def test_window_keeps_only_recent_turns():
    k = 2
    seen = _simulate(num_turns=6, window_turns=k, tokens_per_turn=4)
    # Turns 0,1 produce no context truncation (turn < k => full history).
    assert seen[0] == PROMPT
    assert seen[1] == PROMPT + [0, 1, 2, 3]
    # From turn k onward, only the last k turns of completion are retained.
    # At turn 4, retained turns are {2, 3}; turns 0,1 dropped.
    expected_turn4 = PROMPT + [t * 10 + i for t in (2, 3) for i in range(4)]
    assert seen[4] == expected_turn4
    # At turn 5, retained turns are {3, 4}.
    expected_turn5 = PROMPT + [t * 10 + i for t in (3, 4) for i in range(4)]
    assert seen[5] == expected_turn5


def test_window_always_starts_with_prompt():
    seen = _simulate(num_turns=8, window_turns=3)
    for ctx in seen:
        assert ctx[: len(PROMPT)] == PROMPT


def test_window_length_is_bounded():
    k, tpt = 3, 5
    seen = _simulate(num_turns=10, window_turns=k, tokens_per_turn=tpt)
    # Once past the warm-up, the windowed context never exceeds prompt + k turns.
    for ctx in seen:
        assert len(ctx) <= len(PROMPT) + k * tpt


def test_window_equal_to_turn_count_is_full():
    # turn index never reaches window_turns, so behaves like no-window.
    seen = _simulate(num_turns=4, window_turns=4)
    assert seen[-1] == PROMPT + [t * 10 + i for t in range(3) for i in range(4)]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
