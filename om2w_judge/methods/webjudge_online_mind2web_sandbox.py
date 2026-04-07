import asyncio
import json
import re

from PIL import Image

from om2w_judge.methods.webjudge_online_mind2web import MAX_IMAGE, identify_key_points, judge_image
from om2w_judge.utils import encode_image

IMAGE_PARSE_MAX_RETRIES = 3


def _format_entries(entries):
    if not entries:
        return ""

    formatted = []
    for index, entry in enumerate(entries, start=1):
        if isinstance(entry, str):
            text = entry
        else:
            text = json.dumps(entry, ensure_ascii=False)
        formatted.append(f"{index}. {text}")
    return "\n".join(formatted)


def _parse_image_judge_response(response):
    score_match = re.search(r"(?is)\bscore\b[^1-5]*([1-5])\b", response)
    reasoning_match = re.search(
        r"(?is)(?:\*\*?\s*reasoning\s*\*\*?|reasoning)\s*[:\-]\s*(.*?)(?=\n\s*(?:\d+\.\s*)?(?:\*\*?\s*score\s*\*\*?|score)\s*[:\-]|\Z)",
        response,
    )

    if score_match and reasoning_match:
        reasoning = re.sub(r"\s+", " ", reasoning_match.group(1)).strip()
        return reasoning, int(score_match.group(1))

    try:
        payload = json.loads(response)
    except Exception:
        payload = None

    if isinstance(payload, dict):
        score = payload.get("Score", payload.get("score"))
        reasoning = payload.get("Reasoning", payload.get("reasoning"))
        if isinstance(score, int) and 1 <= score <= 5 and isinstance(reasoning, str) and reasoning.strip():
            return re.sub(r"\s+", " ", reasoning).strip(), score

    raise ValueError("Could not parse image judge response")


async def _judge_image_with_retry(task, image_path, key_points, model):
    last_response = ""
    last_error = None
    for attempt in range(1, IMAGE_PARSE_MAX_RETRIES + 1):
        last_response = await judge_image(task, image_path, key_points, model)
        try:
            reasoning, score = _parse_image_judge_response(last_response)
            return {
                "Response": last_response,
                "Score": score,
                "Reasoning": reasoning,
                "Attempts": attempt,
                "ParseFailed": False,
            }
        except Exception as exc:
            last_error = exc
            print(f"Error processing response on attempt {attempt}/{IMAGE_PARSE_MAX_RETRIES}: {exc}")

    return {
        "Response": last_response,
        "Score": 0,
        "Reasoning": "",
        "Attempts": IMAGE_PARSE_MAX_RETRIES,
        "ParseFailed": True,
        "ParseError": str(last_error) if last_error is not None else "unknown",
    }


async def _webjudge_online_mind2web_sandbox_eval(
    task,
    thoughts,
    last_actions,
    images_path,
    model,
    score_threshold,
    include_action_history,
):
    if include_action_history:
        system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's thoughts, action history, key points for task completion, some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), the task is not considered successful.
2: You must carefully check whether these snapshots and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
3: Certain key points or requirements should be applied by the filter. Otherwise, a search with all requirements as input will be deemed a failure since it cannot guarantee that all results meet the requirements!
4: If the task requires filtering by a specific range of money, years, or the number of beds and bathrooms, the applied filter must exactly match the given requirement. Any deviation results in failure. To ensure the task is successful, the applied filter must precisely match the specified range without being too broad or too narrow.
Examples of Failure Cases:
- If the requirement is less than $50, but the applied filter is less than $25, it is a failure.
- If the requirement is $1500-$2500, but the applied filter is $2000-$2500, it is a failure.
- If the requirement is $25-$200, but the applied filter is $0-$200, it is a failure.
- If the required years are 2004-2012, but the filter applied is 2001-2012, it is a failure.
- If the required years are before 2015, but the applied filter is 2000-2014, it is a failure.
- If the task requires exactly 2 beds, but the filter applied is 2+ beds, it is a failure.
5: Some tasks require a submission action or a display of results to be considered successful.
6: If the retrieved information is invalid or empty(e.g., No match was found), but the agent has correctly performed the required action, it should still be considered successful.
7: When a task specifies multiple explicit filters or sort constraints, the agent must apply each one through the site controls when those controls exist. It is not sufficient to infer compliance from unfiltered or broadly searched results.
8: If a relevant site control does not exist, the trajectory must show direct verification of that constraint from the page content or item details before the task can be marked successful.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"
"""
        prompt = """User Task: {task}

Key Points: {key_points}

Trajectory Thoughts:
{thoughts}

Action History:
{last_actions}

The potentially important snapshots of the webpage in the agent's trajectory and their reasons:
{image_thoughts}"""
        no_image_prompt = """User Task: {task}

Key Points: {key_points}

Trajectory Thoughts:
{thoughts}

Action History:
{last_actions}"""
    else:
        system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's thoughts, key points for task completion, and some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), the task is not considered successful.
2: You must carefully check whether these snapshots meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
3: Certain key points or requirements should be applied by the filter. Otherwise, a search with all requirements as input will be deemed a failure since it cannot guarantee that all results meet the requirements!
4: If the task requires filtering by a specific range of money, years, or the number of beds and bathrooms, the applied filter must exactly match the given requirement. Any deviation results in failure. To ensure the task is successful, the applied filter must precisely match the specified range without being too broad or too narrow.
Examples of Failure Cases:
- If the requirement is less than $50, but the applied filter is less than $25, it is a failure.
- If the requirement is $1500-$2500, but the applied filter is $2000-$2500, it is a failure.
- If the requirement is $25-$200, but the applied filter is $0-$200, it is a failure.
- If the required years are 2004-2012, but the filter applied is 2001-2012, it is a failure.
- If the required years are before 2015, but the applied filter is 2000-2014, it is a failure.
- If the task requires exactly 2 beds, but the filter applied is 2+ beds, it is a failure.
5: Some tasks require a submission action or a display of results to be considered successful.
6: If the retrieved information is invalid or empty(e.g., No match was found), but the agent has correctly performed the required action, it should still be considered successful.
7: When a task specifies multiple explicit filters or sort constraints, the agent must apply each one through the site controls when those controls exist. It is not sufficient to infer compliance from unfiltered or broadly searched results.
8: If a relevant site control does not exist, the trajectory must show direct verification of that constraint from the page content or item details before the task can be marked successful.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"
"""
        prompt = """User Task: {task}

Key Points: {key_points}

Trajectory Thoughts:
{thoughts}

The potentially important snapshots of the webpage in the agent's trajectory and their reasons:
{image_thoughts}"""
        no_image_prompt = """User Task: {task}

Key Points: {key_points}

Trajectory Thoughts:
{thoughts}"""

    key_points = await identify_key_points(task, model)
    key_points = key_points.replace("\n\n", "\n")

    try:
        key_points = key_points.split("**Key Points**:")[1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
    except Exception:
        key_points = key_points.split("Key Points:")[-1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())

    tasks = [_judge_image_with_retry(task, image_path, key_points, model) for image_path in images_path]
    image_results = await asyncio.gather(*tasks)

    whole_content_img = []
    whole_thoughts = []
    record = []
    for image_result, image_path in zip(image_results, images_path):
        image_thought = image_result["Reasoning"]
        score = image_result["Score"]
        record.append(
            {
                "Response": image_result["Response"],
                "Score": score,
                "Attempts": image_result["Attempts"],
                "ParseFailed": image_result["ParseFailed"],
                **({"ParseError": image_result["ParseError"]} if image_result["ParseFailed"] else {}),
            }
        )

        if score >= score_threshold:
            jpg_base64_str = encode_image(Image.open(image_path))
            whole_content_img.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{jpg_base64_str}", "detail": "high"},
                }
            )
            if image_thought != "":
                whole_thoughts.append(image_thought)

    whole_content_img = whole_content_img[:MAX_IMAGE]
    whole_thoughts = whole_thoughts[:MAX_IMAGE]
    if len(whole_content_img) == 0:
        prompt = no_image_prompt

    text = prompt.format(
        task=task,
        thoughts=_format_entries(thoughts),
        last_actions=_format_entries(last_actions),
        key_points=key_points,
        image_thoughts="\n".join(f"{i+1}. {thought}" for i, thought in enumerate(whole_thoughts)),
    )

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [{"type": "text", "text": text}] + whole_content_img,
        },
    ]
    return messages, text, system_msg, record, key_points


async def WebJudge_Online_Mind2Web_Sandbox_eval(task, thoughts, last_actions, images_path, model, score_threshold):
    return await _webjudge_online_mind2web_sandbox_eval(
        task,
        thoughts,
        last_actions,
        images_path,
        model,
        score_threshold,
        include_action_history=True,
    )


async def WebJudge_Online_Mind2Web_Sandbox_ThoughtsOnly_eval(task, thoughts, images_path, model, score_threshold):
    return await _webjudge_online_mind2web_sandbox_eval(
        task,
        thoughts,
        [],
        images_path,
        model,
        score_threshold,
        include_action_history=False,
    )
