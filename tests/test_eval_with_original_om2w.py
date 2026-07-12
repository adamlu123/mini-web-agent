from scripts.eval_with_original_om2w import bound_action_history


def test_bound_action_history_preserves_small_history() -> None:
    actions = ["open page", "apply filter", "verify results"]

    assert bound_action_history(actions) is actions


def test_bound_action_history_keeps_both_ends_within_limits() -> None:
    actions = [f"action-{index}-" + ("x" * 500) for index in range(1_000)]

    bounded = bound_action_history(actions)

    assert len(bounded) == 501
    assert bounded[0].startswith("action-0-")
    assert "500 action log line(s) omitted" in bounded[250]
    assert bounded[-1].startswith("action-999-")
    assert sum(map(len, bounded)) <= 60_000
