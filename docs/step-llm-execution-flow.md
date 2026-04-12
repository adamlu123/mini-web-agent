# Per-step LLM Execution Flow

This document explains one agent step end to end: what goes into the LLM request, how the model output is parsed, what stays local-only, and how the resulting observation becomes the next model input.

## 1. ASCII diagram

```text
+------------------------------------------------------------------------------------------------+
| Step N starts in DefaultAgent.step()                                                           |
| Code: agents/default.py                                                                        |
+------------------------------------------------------------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------------------+
| 1. Agent has full local history in self.messages                                               |
|                                                                                                |
|    self.messages = [                                                                           |
|      system message,                                                                           |
|      initial user/task message,                                                                |
|      assistant thought from step 1,                                                            |
|      observation user message from step 1,                                                     |
|      assistant thought from step 2,                                                            |
|      observation user message from step 2,                                                     |
|      ...                                                                                       |
|    ]                                                                                           |
+------------------------------------------------------------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------------------+
| 2. DefaultAgent.query() calls model.query(self.messages)                                       |
| Code: agents/default.py                                                                        |
+------------------------------------------------------------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------------------+
| 3. PhyagiModel._query_async(messages) clones history into request_messages                     |
|                                                                                                |
|    request_messages = list(messages)                                                           |
|                                                                                                |
|    Important: this still contains full local message objects with:                             |
|      - role                                                                                    |
|      - content                                                                                 |
|      - extra                                                                                   |
|                                                                                                |
|    But only role/content survive the next serialization step.                                  |
+------------------------------------------------------------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------------------+
| 4. _build_payload(request_messages)                                                            |
|    -> _serialize_response_input(messages)                                                      |
|                                                                                                |
|    Serialized into gateway input messages with only:                                           |
|      - role                                                                                    |
|      - content text parts                                                                      |
|      - content image parts                                                                     |
|                                                                                                |
|    Not sent to the LLM:                                                                        |
|      - message.extra.actions                                                                   |
|      - message.extra.done                                                                      |
|      - message.extra.final_response                                                            |
|      - message.extra.raw_response                                                              |
|      - message.extra.usage                                                                     |
+------------------------------------------------------------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------------------+
| 5. Gateway / LLM call                                                                          |
|                                                                                                |
|    Sent to the LLM for step N:                                                                 |
|      - rendered system prompt                                                                  |
|      - rendered task/instance prompt                                                           |
|      - previous assistant thought texts                                                        |
|      - previous observation messages                                                           |
|      - observation screenshots, if attach_observation_screenshot=true                          |
|      - format-repair user message, only if a previous parse attempt failed                     |
+------------------------------------------------------------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------------------+
| 6. Raw model output comes back                                                                 |
| Code: models/phyagi_model.py                                                                   |
|                                                                                                |
|    raw_text = _extract_response_text(response_payload)                                         |
|                                                                                                |
|    Then parsed as either:                                                                      |
|      - XML  -> parse_xml_output(raw_text)                                                      |
|      - JSON -> parse_json_output(raw_text)                                                     |
+------------------------------------------------------------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------------------+
| 7. Parsed response normalized into local assistant metadata                                    |
|                                                                                                |
|    parsed = {                                                                                  |
|      thought,                                                                                  |
|      bash_command or python_code,                                                              |
|      done,                                                                                     |
|      final_response                                                                            |
|    }                                                                                           |
|                                                                                                |
|    actions = []                                                                                |
|    if bash_command: actions = [{bash_command, command}]                                        |
|    elif python_code: actions = [{python_code}]                                                 |
|                                                                                                |
|    assistant_message = {                                                                       |
|      role: "assistant",                                                                       |
|      content: parsed.thought,                                                                  |
|      extra: {                                                                                  |
|        actions,                                                                                |
|        done,                                                                                   |
|        final_response,                                                                         |
|        raw_response: parsed,                                                                   |
|        usage                                                                                   |
|      }                                                                                         |
|    }                                                                                           |
+------------------------------------------------------------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------------------+
| 8. DefaultAgent adds that assistant message to self.messages                                   |
|                                                                                                |
|    This means the next LLM call will see the assistant thought in content,                     |
|    but not the extra metadata directly.                                                        |
+------------------------------------------------------------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------------------+
| 9a. If done=true                                                                               |
|                                                                                                |
|    - no environment action runs                                                                |
|    - agent emits an exit message                                                               |
|    - loop ends                                                                                 |
+------------------------------------------------------------------------------------------------+
                                         |
                                         |
                                         +-------------------------------------------------------+
                                                                                                 |
                                         v                                                       |
+------------------------------------------------------------------------------------------------+|
| 9b. If done=false                                                                              ||
|                                                                                                ||
|    DefaultAgent.execute_actions() runs env.execute(action) for each action                     ||
|                                                                                                ||
|    Local workspace mode: executes shell command, captures output/log/screenshot metadata       ||
|    Local browser mode: runs Python Playwright code, captures URL/title/console/ARIA/screenshot ||
+------------------------------------------------------------------------------------------------+|
                                         |                                                       |
                                         v                                                       |
+------------------------------------------------------------------------------------------------+|
| 10. Model.format_observation_messages(...) converts outputs into new user messages             ||
|                                                                                                ||
|     Each observation message contains:                                                         ||
|       - text observation summary rendered from observation_template                            ||
|       - optional screenshot image part built from screenshot_path                             ||
|       - the path string itself may also appear in the text summary                            ||
|                                                                                                ||
|     These observation user messages are appended to self.messages.                             ||
+------------------------------------------------------------------------------------------------+|
                                         |                                                       |
                                         +------------------------ back to step N+1 query -------+
```

## 2. What is actually fed into the LLM

For step `N`, the LLM receives a serialized form of `request_messages`, not the raw Python dicts.

Conceptually the input looks like this:

```text
request_messages (local Python objects)
  = [
      {role: system, content: rendered system prompt, extra: {...}},
      {role: user, content: rendered task prompt, extra: {...}},
      {role: assistant, content: previous thought, extra: {...}},
      {role: user, content: observation text [+ screenshot], extra: {...}},
      ...
    ]

_serialize_response_input(request_messages)
  -> [
       {type: message, role: developer, content: [...]},
       {type: message, role: user,      content: [...]},
       {type: message, role: assistant, content: [...]},
       {type: message, role: user,      content: [...]},
       ...
     ]
```

What is preserved:

- `role`
- `content` text
- `content` image attachments for observations when enabled

Screenshot detail:

- if an observation has `screenshot_path`, `format_observation_messages` uses that path to read the file and append an `input_image` part
- the default observation template also renders `Screenshot path: ...` into the text body, so the path string itself can be visible to the model as plain text
- the image attachment is not created from `extra`; it is created from `content`

What is dropped before the gateway call:

- `extra.actions`
- `extra.done`
- `extra.final_response`
- `extra.raw_response`
- `extra.usage`
- any other `extra` keys

## 3. Parse failure and retry path

If the model output cannot be parsed as the expected XML or JSON shape:

```text
raw_text
  -> parse fails
  -> _format_repair_message(raw_text, error)
  -> append one new user message to request_messages
  -> rebuild payload
  -> send another LLM request
```

Important detail:

- the repair message is appended to `request_messages`, not to a fresh empty history
- so the retry still contains the full prior conversation plus one extra corrective user prompt

## 4. How previous history becomes next input

The transition from one step to the next is:

```text
assistant step result
  -> appended to self.messages as an assistant message
  -> action executed in env
  -> env returns observation
  -> observation converted into one or more user messages
  -> those user messages appended to self.messages
  -> next model.query(self.messages)
```

So the next LLM input is built from:

1. all earlier assistant thought texts
2. all earlier observation user messages
3. the original system and task prompts
4. any format-repair message added during parse retries

## 5. Local-only metadata vs model-visible content

This distinction is easy to miss.

Model-visible:

- `message["content"]`
- rendered observation text
- attached observation screenshots
- any screenshot path text that was rendered into the observation text template

Local-only runtime metadata:

- `message["extra"]`
- normalized `actions`
- `done`
- `final_response`
- `raw_response`
- `usage`

The local-only metadata drives agent control flow, debug artifact writing, evaluation export, and trace viewing, but it is not directly sent back into the next LLM call.