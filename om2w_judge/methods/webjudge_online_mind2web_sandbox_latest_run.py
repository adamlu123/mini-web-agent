"""Sandbox judge wrappers for latest-run-folder compatible evaluation.

The judging prompt logic is unchanged from the legacy sandbox evaluator.
Artifact selection now happens before these functions are called so the
evaluator can stay focused on the judging prompt itself.
"""

from om2w_judge.methods.webjudge_online_mind2web_sandbox import (
    WebJudge_Online_Mind2Web_Sandbox_ThoughtsOnly_eval as _legacy_thoughts_only_eval,
    WebJudge_Online_Mind2Web_Sandbox_WithThoughts_eval as _legacy_with_thoughts_eval,
    WebJudge_Online_Mind2Web_Sandbox_eval as _legacy_sandbox_eval,
)


async def WebJudge_Online_Mind2Web_Sandbox_eval(task, thoughts, last_actions, images_path, model, score_threshold):
    return await _legacy_sandbox_eval(task, thoughts, last_actions, images_path, model, score_threshold)


async def WebJudge_Online_Mind2Web_Sandbox_WithThoughts_eval(
    task,
    thoughts,
    last_actions,
    images_path,
    model,
    score_threshold,
):
    return await _legacy_with_thoughts_eval(task, thoughts, last_actions, images_path, model, score_threshold)


async def WebJudge_Online_Mind2Web_Sandbox_ThoughtsOnly_eval(task, thoughts, images_path, model, score_threshold):
    return await _legacy_thoughts_only_eval(task, thoughts, images_path, model, score_threshold)
