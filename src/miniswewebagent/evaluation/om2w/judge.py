"""Online-Mind2Web judge implementation used by mini-web-agent evaluators.

Originally this implementation was embedded in an executable under
``scripts/archive``. It lives in the package so active evaluators never import
archived code.
"""

from __future__ import annotations

import asyncio
import json
import re

from PIL import Image

from om2w_judge.methods import webjudge_online_mind2web as upstream_webjudge

IMAGE_PARSE_MAX_RETRIES = 10
RETRYABLE_STATUS_CODES = frozenset({400, 408, 409, 425, 429, 500, 502, 503, 504})
RETRYABLE_STATUS_CODE_TEXT = frozenset(str(code) for code in RETRYABLE_STATUS_CODES)
RETRYABLE_RESPONSE_TEXT_MARKERS = (
    "rate limit",
    "ratelimit",
    "too many requests",
    "bad gateway",
    "gateway timeout",
    "temporarily unavailable",
    "service unavailable",
    "server disconnected",
    "connection reset",
    "timed out",
)


def parse_image_judge_response(response: str) -> tuple[str, int]:
    score_match = re.search(r"(?is)\bscore\b[^1-5]*([1-5])\b", response)
    reasoning_match = re.search(
        r"(?is)(?:\*\*?\s*reasoning\s*\*\*?|reasoning)\s*[:\-]\s*"
        r"(.*?)(?=\n\s*(?:\d+\.\s*)?(?:\*\*?\s*score\s*\*\*?|score)\s*[:\-]|\Z)",
        response,
    )
    if score_match and reasoning_match:
        reasoning = re.sub(r"\s+", " ", reasoning_match.group(1)).strip()
        return reasoning, int(score_match.group(1))

    try:
        payload = json.loads(response)
    except (TypeError, ValueError):
        payload = None

    if isinstance(payload, dict):
        score = payload.get("Score", payload.get("score"))
        reasoning = payload.get("Reasoning", payload.get("reasoning"))
        if isinstance(score, str) and score.strip().isdigit():
            score = int(score.strip())
        if (
            isinstance(score, int)
            and 1 <= score <= 5
            and isinstance(reasoning, str)
            and reasoning.strip()
        ):
            return re.sub(r"\s+", " ", reasoning).strip(), score

    raise ValueError("Could not parse image judge response")


def retryable_image_judge_response_error(response: str) -> str | None:
    text = response.strip()
    if not text:
        return "empty image judge response"

    lowered = text.lower()
    if text in RETRYABLE_STATUS_CODE_TEXT:
        return f"retryable gateway status text: {text}"
    for marker in RETRYABLE_RESPONSE_TEXT_MARKERS:
        if marker in lowered:
            snippet = re.sub(r"\s+", " ", text)[:160]
            return f"retryable gateway response text: {snippet}"

    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None

    status_candidates = (
        payload.get("status"),
        payload.get("status_code"),
        payload.get("code"),
    )
    for candidate in status_candidates:
        if isinstance(candidate, str) and candidate.strip().isdigit():
            candidate = int(candidate.strip())
        if isinstance(candidate, int) and candidate in RETRYABLE_STATUS_CODES:
            return f"retryable gateway response payload status: {candidate}"

    error_payload = payload.get("error")
    if isinstance(error_payload, dict):
        code = error_payload.get("code")
        if isinstance(code, str) and code.strip().isdigit():
            code = int(code.strip())
        if isinstance(code, int) and code in RETRYABLE_STATUS_CODES:
            return f"retryable gateway response error code: {code}"
        message = error_payload.get("message")
        if isinstance(message, str):
            lowered_message = message.lower()
            if any(marker in lowered_message for marker in RETRYABLE_RESPONSE_TEXT_MARKERS):
                snippet = re.sub(r"\s+", " ", message)[:160]
                return f"retryable gateway response error: {snippet}"

    message = payload.get("message")
    if isinstance(message, str):
        lowered_message = message.lower()
        if any(marker in lowered_message for marker in RETRYABLE_RESPONSE_TEXT_MARKERS):
            snippet = re.sub(r"\s+", " ", message)[:160]
            return f"retryable gateway response message: {snippet}"

    return None


async def judge_image_with_retry(
    task: str,
    image_path: str,
    key_points: str,
    model,
) -> dict[str, object]:
    last_response = ""
    last_error: BaseException | None = None
    for attempt in range(1, IMAGE_PARSE_MAX_RETRIES + 1):
        try:
            last_response = await upstream_webjudge.judge_image(
                task,
                image_path,
                key_points,
                model,
            )
        except Exception as exc:  # noqa: BLE001 - retry the upstream gateway boundary
            last_error = exc
            print(
                f"Error loading/judging image on attempt {attempt}/"
                f"{IMAGE_PARSE_MAX_RETRIES} for {image_path}: {exc}"
            )
            continue

        # Parse before checking gateway markers: valid reasoning may itself
        # mention phrases such as "timed out".
        try:
            reasoning, score = parse_image_judge_response(last_response)
            return {
                "Response": last_response,
                "Score": score,
                "Reasoning": reasoning,
                "Attempts": attempt,
                "ParseFailed": False,
            }
        except ValueError as parse_exc:
            retryable_error = retryable_image_judge_response_error(last_response)
            if retryable_error is not None:
                last_error = RuntimeError(retryable_error)
                print(
                    f"Error processing response on attempt {attempt}/"
                    f"{IMAGE_PARSE_MAX_RETRIES}: {retryable_error}"
                )
                continue
            last_error = parse_exc
            print(
                f"Error processing response on attempt {attempt}/"
                f"{IMAGE_PARSE_MAX_RETRIES}: {parse_exc}"
            )

    return {
        "Response": last_response,
        "Score": 0,
        "Reasoning": "",
        "Attempts": IMAGE_PARSE_MAX_RETRIES,
        "ParseFailed": True,
        "ParseError": str(last_error) if last_error is not None else "unknown",
    }


async def robust_webjudge_online_mind2web_eval(
    task,
    last_actions,
    images_path,
    model,
    score_threshold,
):
    """Build the final WebJudge request with retry-hardened image scoring."""
    # Keep this adapter-local copy aligned with the official OSU implementation.
    # Tests compare the complete requests produced by both evaluator paths.
    system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's action history, key points for task completion, some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

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
7: If the current page already displays all available items, then applying a filter is not necessary. As long as the agent selects items that meet the requirements (e.g., the cheapest or lowest price), the task is still considered successful.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: \"success\" or \"failure\"
"""
    prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}

The potentially important snapshots of the webpage in the agent's trajectory and their reasons:
{thoughts}"""

    key_points = await upstream_webjudge.identify_key_points(task, model)
    key_points = key_points.replace("\n\n", "\n")
    try:
        key_points = key_points.split("**Key Points**:")[1]
    except IndexError:
        key_points = key_points.split("Key Points:")[-1]
    key_points = "\n".join(line.lstrip() for line in key_points.splitlines())

    image_records = await asyncio.gather(
        *[judge_image_with_retry(task, image_path, key_points, model) for image_path in images_path]
    )

    whole_content_img = []
    whole_thoughts = []
    record = []
    for image_record, image_path in zip(image_records, images_path):
        record.append(image_record)
        score = int(image_record.get("Score", 0) or 0)
        thought = str(image_record.get("Reasoning", "") or "").strip()
        if score >= score_threshold:
            jpg_base64_str = upstream_webjudge.encode_image(Image.open(image_path))
            whole_content_img.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{jpg_base64_str}",
                        "detail": "high",
                    },
                }
            )
            if thought:
                whole_thoughts.append(thought)

    whole_content_img = whole_content_img[: upstream_webjudge.MAX_IMAGE]
    whole_thoughts = whole_thoughts[: upstream_webjudge.MAX_IMAGE]
    if not whole_content_img:
        prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}"""

    text = prompt.format(
        task=task,
        last_actions="\n".join(
            f"{index + 1}. {action}" for index, action in enumerate(last_actions)
        ),
        key_points=key_points,
        thoughts="\n".join(
            f"{index + 1}. {thought}" for index, thought in enumerate(whole_thoughts)
        ),
    )
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [{"type": "text", "text": text}] + whole_content_img,
        },
    ]
    return messages, text, system_msg, record, key_points
