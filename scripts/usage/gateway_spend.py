#!/usr/bin/env python3
"""Report Phi gateway spend from Application Insights.

The gateway writes per-request cost to Postgres (`request_metrics`) and mirrors the
same fields to the `api-gateway-vmss-monitor` Application Insights component, which is
readable with a plain `az` login. Telemetry carries `user_id` but no `api_key_id`, so
spend is attributed per user; `--key` resolves a raw gateway key to its owner first.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

APP = "api-gateway-vmss-monitor"
RESOURCE_GROUP = "services"
SUBSCRIPTION = "2aac527a-de5a-4fe3-95e9-5c8b9d48ed62"

GATEWAY_CHAT_URL = "http://gateway.phyagi.net/api/chat/completions"
PROBE_MODEL = "gpt-5.2"
PROBE_TIMEOUT_S = 300
PROBE_POLL_S = 20


def _az_query(kql: str, hours: int) -> list[dict[str, object]]:
    """Run a KQL query against the gateway Application Insights component.

    Args:
        kql: Query body, without a timespan filter.
        hours: Lookback window in hours, passed as the CLI timespan.

    Returns:
        Result rows as dicts keyed by column name.

    Raises:
        SystemExit: If the `az` call fails.
    """

    # `az monitor app-insights query` defaults to a 1h timespan and INTERSECTS it with
    # any `ago()` in the query, so the window must be set here, not in the KQL.
    command = [
        "az", "monitor", "app-insights", "query",
        "--app", APP,
        "-g", RESOURCE_GROUP,
        "--subscription", SUBSCRIPTION,
        "--offset", f"{hours}h",
        "--analytics-query", kql,
        "-o", "json",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"az query failed:\n{result.stderr.strip()}")

    table = json.loads(result.stdout)["tables"][0]
    columns = [column["name"] for column in table["columns"]]
    return [dict(zip(columns, row)) for row in table["rows"]]


def identify_key(api_key: str) -> tuple[str, str, str]:
    """Resolve a gateway API key to its owning identity.

    Sends one deliberately over-constrained request, which the upstream rejects before
    generating any billable tokens but which the gateway still logs with caller dims.

    Args:
        api_key: Raw gateway API key.

    Returns:
        Tuple of user name, user id, and team name.

    Raises:
        SystemExit: If the key is rejected or no telemetry appears in time.
    """

    session_id = f"gateway-spend-probe-{uuid.uuid4().hex[:12]}"
    payload = json.dumps({
        "model": PROBE_MODEL,
        "session_id": session_id,
        "max_completion_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }).encode()
    request = urllib.request.Request(
        GATEWAY_CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )

    try:
        urllib.request.urlopen(request, timeout=120).read()
    except urllib.error.HTTPError as error:
        body = json.loads(error.read() or b"{}").get("error", {})
        if body.get("code") in {"auth.required", "auth.api_key_unknown", "auth.api_key_revoked"}:
            sys.exit(f"Key not accepted by the gateway: {body.get('code')}")
    except urllib.error.URLError as error:
        sys.exit(f"Gateway unreachable: {error}")

    print(f"probing key identity (session {session_id}), waiting for telemetry...", file=sys.stderr)
    deadline = time.time() + PROBE_TIMEOUT_S
    while time.time() < deadline:
        rows = _az_query(
            f"requests"
            f"| where tostring(customDimensions.session_id) == '{session_id}'"
            f"| project user=tostring(customDimensions.user_name),"
            f" uid=tostring(customDimensions.user_id), team=tostring(customDimensions.team_name)"
            f"| take 1",
            hours=1,
        )
        if rows:
            row = rows[0]
            return str(row["user"]), str(row["uid"]), str(row["team"])
        time.sleep(PROBE_POLL_S)

    sys.exit("Key accepted but no telemetry surfaced; retry or pass --user explicitly.")


def spend_by_model(user_id: str, hours: int) -> list[dict[str, object]]:
    """Return one user's spend grouped by model.

    Args:
        user_id: Gateway user id.
        hours: Lookback window in hours.

    Returns:
        Rows of model, requests, tokens, and spend.
    """

    return _az_query(
        f"requests"
        f"| where name == 'completions' and tostring(customDimensions.user_id) == '{user_id}'"
        f"| summarize requests=count(), tokens=sum(tolong(customDimensions.total_tokens)),"
        f" spend_usd=sum(todouble(customDimensions.spend_usd))"
        f" by model=tostring(customDimensions.model)"
        f"| order by spend_usd desc",
        hours,
    )


def spend_by_user(team: str, hours: int) -> list[dict[str, object]]:
    """Return a team's spend grouped by user.

    Args:
        team: Team name as recorded in telemetry.
        hours: Lookback window in hours.

    Returns:
        Rows of user, requests, tokens, spend, and models used.
    """

    return _az_query(
        f"requests"
        f"| where name == 'completions' and tostring(customDimensions.team_name) == '{team}'"
        f"| summarize requests=count(), tokens=sum(tolong(customDimensions.total_tokens)),"
        f" spend_usd=sum(todouble(customDimensions.spend_usd)),"
        f" models=strcat_array(array_sort_asc(make_set(tostring(customDimensions.model))), ' ')"
        f" by user=tostring(customDimensions.user_name)"
        f"| order by spend_usd desc",
        hours,
    )


def hourly_series(where: str, hours: int) -> list[dict[str, object]]:
    """Return an hourly spend series for a telemetry subset.

    Args:
        where: KQL predicate selecting the subset.
        hours: Lookback window in hours.

    Returns:
        Rows of hour bucket, requests, and spend.
    """

    return _az_query(
        f"requests"
        f"| where name == 'completions' and {where}"
        f"| summarize requests=count(), spend_usd=sum(todouble(customDimensions.spend_usd))"
        f" by hour=bin(timestamp, 1h)"
        f"| order by hour asc",
        hours,
    )


def render(rows: list[dict[str, object]], label: str) -> None:
    """Print a spend table with a total row.

    Args:
        rows: Query rows whose first column is the grouping key.
        label: Header for the grouping column.
    """

    if not rows:
        print("no requests in window")
        return

    key = next(iter(rows[0]))
    width = max(len(label), *(len(str(row[key])) for row in rows))
    print(f"{label:<{width}}  {'requests':>9}  {'tokens':>16}  {'spend_usd':>11}")
    for row in rows:
        print(
            f"{str(row[key]):<{width}}  {row['requests']:>9,}  "
            f"{int(row['tokens'] or 0):>16,}  {float(row['spend_usd'] or 0):>11.2f}"
        )
    print("-" * (width + 44))
    print(
        f"{'TOTAL':<{width}}  {sum(int(r['requests']) for r in rows):>9,}  "
        f"{sum(int(r['tokens'] or 0) for r in rows):>16,}  "
        f"{sum(float(r['spend_usd'] or 0) for r in rows):>11.2f}"
    )


def main() -> None:
    """Parse arguments and print the requested spend report."""

    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--key", help="gateway API key to attribute (default: $OPENAI_GATEWAY_API_KEY)"
    )
    target.add_argument("--user", help="user name, skipping key identification")
    target.add_argument("--team", help="team name; reports spend per user")
    parser.add_argument("--hours", type=int, default=24, help="lookback window (default: 24)")
    parser.add_argument("--hourly", action="store_true", help="also print the hourly series")
    parser.add_argument("--json", action="store_true", help="emit raw rows as JSON")
    args = parser.parse_args()

    if args.team:
        rows = spend_by_user(args.team, args.hours)
        where = f"tostring(customDimensions.team_name) == '{args.team}'"
        heading, label = f"team {args.team}", "user"
    else:
        if args.user:
            where = f"tostring(customDimensions.user_name) == '{args.user}'"
            rows = _az_query(
                f"requests | where name == 'completions' and {where}"
                f"| summarize requests=count(), tokens=sum(tolong(customDimensions.total_tokens)),"
                f" spend_usd=sum(todouble(customDimensions.spend_usd))"
                f" by model=tostring(customDimensions.model) | order by spend_usd desc",
                args.hours,
            )
            heading = f"user {args.user}"
        else:
            api_key = args.key or os.environ.get("OPENAI_GATEWAY_API_KEY")
            if not api_key:
                parser.error("no --key/--user/--team and $OPENAI_GATEWAY_API_KEY is unset")
            user, user_id, team = identify_key(api_key)
            rows = spend_by_model(user_id, args.hours)
            where = f"tostring(customDimensions.user_id) == '{user_id}'"
            heading = f"user {user} (team {team})"
        label = "model"

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print(f"\n{heading} — last {args.hours}h\n")
    render(rows, label)

    if args.hourly:
        print("\nhourly")
        for row in hourly_series(where, args.hours):
            spend = float(row["spend_usd"] or 0)
            print(f"  {str(row['hour'])[:13]}h  {row['requests']:>7,} req  ${spend:>9.2f}")


if __name__ == "__main__":
    main()
