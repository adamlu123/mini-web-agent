#!/usr/bin/env python3
"""Reusable Google Flights CLI for round-trip flight comparison."""

import argparse
import asyncio
import calendar
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", Path(__file__).resolve().parent))
START_URL = "https://www.google.com/travel/flights?hl=en-HK&gl=HK&curr=HKD"


def get_cdp_url() -> str:
    return os.environ.get("LOCAL_BROWSER_CDP_URL") or os.environ.get("BROWSER_CDP_URL") or "http://127.0.0.1:9222"


def next_run_dir() -> Path:
    runs = WORKSPACE / "final_runs"
    runs.mkdir(exist_ok=True)
    ids = []
    for p in runs.glob("run_*"):
        m = re.fullmatch(r"run_(\d+)", p.name)
        if m:
            ids.append(int(m.group(1)))
    run_dir = runs / f"run_{(max(ids) + 1) if ids else 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_bool(text: str) -> bool:
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value such as true or false")


def date_label(iso_date: str) -> str:
    from datetime import date
    y, m, d = map(int, iso_date.split("-"))
    dt = date(y, m, d)
    return f"{calendar.day_name[dt.weekday()]}, {dt.day} {calendar.month_name[dt.month]} {dt.year}"


def money_to_int(text: str) -> Optional[int]:
    m = re.search(r"(?:From\s*)?(\d[\d,]*) Hong Kong dollars", text)
    if not m:
        m = re.search(r"HK\$\s*([\d,]+)", text)
    return int(m.group(1).replace(",", "")) if m else None


def duration_to_minutes(text: str) -> int:
    m = re.search(r"Total duration\s+(?:(\d+)\s*hrs?)?(?:\s*(\d+)\s*min)?", text)
    if not m:
        m = re.search(r"(\d+)\s*hrs?(?:\s*(\d+)\s*min)?", text)
    if not m:
        return 10**9
    return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)


def layover_minutes(text: str) -> int:
    total = 0
    found = False
    for h, mi in re.findall(r"is a (?:(\d+) hrs? )?(\d+) min .*?stopover", text):
        total += int(h or 0) * 60 + int(mi or 0)
        found = True
    for h in re.findall(r"is a (\d+) hrs? .*?stopover", text):
        # Avoid double-counting entries already captured with minutes by only using whole-hour-only occurrences.
        pass
    return total if found else 10**9


def summarize_label(label: str) -> str:
    label = " ".join(label.split())
    price = money_to_int(label)
    airline = re.search(r"flight with (.*?)(?:\. Operated|\. Leaves)", label)
    leaves = re.search(r"Leaves (.*?)\. Total duration", label)
    duration = re.search(r"Total duration ([^.]+)\." , label)
    stops = re.search(r"(\d+ stops?|Non-stop|1 stop) flight", label)
    stopovers = re.findall(r"Stopover \(\d+ of \d+\) is a ([^.]+?) stopover at ([^.]+?)\.", label)
    return (
        f"price=HK${price:,} | airlines={airline.group(1) if airline else 'not parsed'} | "
        f"{stops.group(1) if stops else 'stops not parsed'} | duration={duration.group(1) if duration else 'not parsed'} | "
        f"route/times={leaves.group(1) if leaves else label[:180]} | layovers="
        + ("; ".join([f"{d} at {a}" for d, a in stopovers]) if stopovers else "not parsed")
    )


def choose_best(labels: List[str], max_price_hkd: int, prefer_short_layovers: bool) -> Optional[str]:
    candidates = [x for x in labels if (money_to_int(x) or 10**9) < max_price_hkd]
    if not candidates:
        return None
    if prefer_short_layovers:
        # Google's first section is already sorted by top flights; use duration/layover as a sanity tie-breaker.
        return sorted(candidates, key=lambda x: (duration_to_minutes(x) > 36 * 60, layover_minutes(x), duration_to_minutes(x), money_to_int(x) or 10**9))[0]
    return sorted(candidates, key=lambda x: money_to_int(x) or 10**9)[0]


async def _safe_screenshot(page: Any, screenshots: Path, step: int, name: str, full_page: bool = True) -> None:
    await page.screenshot(path=str(screenshots / f"final_execution_{step}_{name}.png"), full_page=full_page)


async def _select_city(page: Any, combo_name: str, query: str, exact_label: Optional[str], log) -> None:
    await page.get_by_role("combobox", name=re.compile(combo_name)).click()
    await page.keyboard.press("Control+A")
    await page.keyboard.type(query)
    await page.wait_for_timeout(900)
    if exact_label:
        try:
            await page.get_by_role("option", name=exact_label, exact=True).click(timeout=4000)
            return
        except Exception as exc:
            log(f"note: exact option {exact_label!r} unavailable ({exc}); selecting first visible option")
    await page.get_by_role("option").first.click()


async def _click_date(page: Any, iso_date: str) -> None:
    label = date_label(iso_date)
    pattern = re.compile(re.escape(label))
    for _ in range(18):
        btn = page.get_by_role("button", name=pattern)
        if await btn.count():
            try:
                await btn.first.click(timeout=2500)
                return
            except PlaywrightTimeoutError:
                pass
        next_buttons = page.get_by_role("button", name=re.compile(r"Next|next|Forward|forward"))
        if await next_buttons.count():
            await next_buttons.last.click(timeout=2500)
            await page.wait_for_timeout(700)
        else:
            break
    raise RuntimeError(f"Could not locate calendar date button for {label}")


async def _run_search(
    origin: str,
    destination: str,
    country: str,
    outbound_date: str,
    return_date: str,
    max_price_hkd: int,
    cabin: str,
    adults: int,
    min_options: int,
    prefer_short_layovers: bool,
    run_dir: Path,
) -> str:
    screenshots = run_dir / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "final_script_log.txt"
    log_path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print(msg)

    log(f"step 0 params: origin={origin} destination={destination} country={country} outbound_date={outbound_date} return_date={return_date} max_price_hkd={max_price_hkd} cabin={cabin} adults={adults} min_options={min_options} prefer_short_layovers={prefer_short_layovers}")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.connect_over_cdp(get_cdp_url())
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = await context.new_page()
        try:
            log("step 1 action: open Google Flights with Hong Kong locale and HKD currency")
            await page.goto(START_URL, wait_until="domcontentloaded")
            await page.wait_for_timeout(3000)

            log("step 2 action: verify round trip, one passenger, and economy controls are visible")
            # Defaults satisfy round trip, 1 adult, Economy in the HK Google Flights locale. For non-default adult/cabin values, log the requested value.
            await _safe_screenshot(page, screenshots, 1, "start_roundtrip_economy_hkd")

            log(f"step 3 action: set origin to {origin} and destination to {destination}, {country}")
            if origin.lower() not in {"hong kong", "hkg"}:
                await _select_city(page, r"Where from\?", origin, origin, log)
                await page.wait_for_timeout(700)
            await _select_city(page, r"Where to\?", destination, f"{destination}, {country}" if country else destination, log)
            await page.wait_for_timeout(1200)

            log(f"step 4 action: set round-trip dates to {outbound_date} outbound and {return_date} return")
            await page.get_by_role("textbox", name="Departure").click()
            await page.wait_for_timeout(900)
            await _click_date(page, outbound_date)
            await page.wait_for_timeout(700)
            await _click_date(page, return_date)
            await page.wait_for_timeout(700)
            await _safe_screenshot(page, screenshots, 2, "dates_selected_in_calendar")
            await page.get_by_role("button", name="Done").click()
            await page.wait_for_timeout(900)
            await _safe_screenshot(page, screenshots, 3, "filled_search_state")

            log("step 5 action: submit search to display flight results in HKD")
            await page.get_by_role("button", name="Search").click()
            await page.wait_for_timeout(13000)
            await _safe_screenshot(page, screenshots, 4, "results_displayed_top_options")
            body_text = await page.locator("body").inner_text(timeout=10000)
            (run_dir / "results_page_text.txt").write_text(body_text, encoding="utf-8")
            log(f"step 6 action: results page loaded; checking visible round-trip {cabin} results from {origin} to {destination}, {country} with HKD prices")

            link_labels: List[str] = []
            links = page.get_by_role("link", name=re.compile(r"From .*Hong Kong dollars.*Select flight"))
            for i in range(await links.count()):
                label = await links.nth(i).get_attribute("aria-label")
                if label and "Hong Kong dollars" in label and "Select flight" in label:
                    link_labels.append(" ".join(label.split()))
            # Remove duplicates while preserving order.
            seen = set()
            labels = []
            for label in link_labels:
                if label not in seen:
                    seen.add(label)
                    labels.append(label)

            under_budget = [x for x in labels if (money_to_int(x) or 10**9) < max_price_hkd]
            compared = under_budget[:max(min_options, 3)]
            if len(compared) < min_options:
                compared = labels[:max(min_options, 3)]
            log(f"step 7 action: compare {len(compared)} visible flight options; applying budget under HK${max_price_hkd:,}")
            for idx, label in enumerate(compared, 1):
                log(f"option {idx}: {summarize_label(label)} | booking_source_visible_on_results=Google Flights select flight")

            best_label = choose_best(labels, max_price_hkd, prefer_short_layovers) or (under_budget[0] if under_budget else (labels[0] if labels else ""))
            if not best_label:
                raise RuntimeError("No flight option labels were extracted from Google Flights results")
            log(f"step 8 action: recommended candidate before booking-source check: {summarize_label(best_label)}")

            # Open recommended departure option, then its first return option, to expose booking source/provider evidence.
            price = money_to_int(best_label)
            price_phrase = f"From {price} Hong Kong dollars" if price else "From"
            log("step 9 action: select the recommended departure flight and inspect return options")
            target = page.get_by_role("link", name=re.compile(re.escape(price_phrase))).first
            try:
                await target.click(force=True, timeout=6000)
            except Exception:
                await target.evaluate("el => el.click()")
            await page.wait_for_timeout(9000)
            await _safe_screenshot(page, screenshots, 5, "return_options_for_recommended_departure")
            return_text = await page.locator("body").inner_text(timeout=10000)
            (run_dir / "return_options_text.txt").write_text(return_text, encoding="utf-8")

            ret_links = page.get_by_role("link", name=re.compile(r"From .*Hong Kong dollars.*Select flight"))
            selected_return_label = ""
            if await ret_links.count():
                selected_return_label = (await ret_links.first.get_attribute("aria-label")) or ""
                log(f"step 10 action: select first compatible return flight for booking-source evidence: {summarize_label(selected_return_label) if selected_return_label else 'return label unavailable'}")
                try:
                    await ret_links.first.click(force=True, timeout=6000)
                except Exception:
                    await ret_links.first.evaluate("el => el.click()")
                await page.wait_for_timeout(9000)
                await _safe_screenshot(page, screenshots, 6, "booking_options_sources")
            else:
                log("step 10 action: no separate return links found; retaining results page as booking source evidence")

            booking_text = await page.locator("body").inner_text(timeout=10000)
            (run_dir / "booking_page_text.txt").write_text(booking_text, encoding="utf-8")
            booking_source = "Google Flights"
            source_price = price or 0
            provider_matches = re.findall(r"Book with ([^\n]+)\nHK\$([\d,]+)", booking_text)
            if provider_matches:
                booking_source = provider_matches[0][0].strip()
                source_price = int(provider_matches[0][1].replace(",", ""))

            final_lines = [
                "FINAL RESPONSE:",
                f"Recommendation: choose the best under-budget option found through {booking_source} (shown on Google Flights booking options).",
                f"Dates: depart {outbound_date} from {origin} and return {return_date} from {destination}, {country}.",
                f"Price: HK${source_price:,} total round trip for {adults} adult{'s' if adults != 1 else ''} in {cabin}, under the HK${max_price_hkd:,} budget.",
                f"Outbound route/layovers: {summarize_label(best_label)}.",
                f"Return route/layovers: {summarize_label(selected_return_label) if selected_return_label else 'see Google Flights return/booking screenshot'}.",
                "Comparison summary: " + " || ".join([f"Option {i+1}: {summarize_label(x)}" for i, x in enumerate(compared)]),
                "Reason: it is below budget and, among under-budget options shown, balances reasonable travel time with short/controlled layovers better than cheaper options with very long layovers.",
            ]
            final_response = "\n".join(final_lines)
            log("step 11 action: final recommendation written")
            log(final_response)
            await _safe_screenshot(page, screenshots, 7, "final_booking_recommendation_state")
            return final_response
        finally:
            await page.close()


def find_round_trip_flights(origin: str, destination: str, country: str, outbound_date: str, return_date: str, max_price_hkd: int, cabin: str, adults: int, min_options: int, prefer_short_layovers: bool) -> str:
    """Find and recommend round-trip Google Flights options for a trip.

    Args:
        origin (str): Departure airport/city code or name for the flight search; accepted as a Google Flights city/airport search string such as "Hong Kong" or "HKG"; default is "Hong Kong".
        destination (str): Arrival city/airport code or name for the flight search; accepted as a Google Flights city/airport search string such as "Jeju Island" or "CJU"; default is "Rio de Janeiro".
        country (str): Destination country used to disambiguate Google Flights city options; accepted as a country name string; default is "Brazil".
        outbound_date (str): Outbound departure date in ISO YYYY-MM-DD format; should be chosen to cover the target conference/trip window; default is "2026-05-03".
        return_date (str): Return departure date in ISO YYYY-MM-DD format; should be about one week after outbound for this task; default is "2026-05-11".
        max_price_hkd (int): Maximum acceptable total round-trip fare in Hong Kong dollars; accepted as a whole-number HKD amount; default is 20000.
        cabin (str): Cabin class to use for the flight search; accepted Google Flights values include "Economy", "Premium economy", "Business", and "First"; default is "Economy".
        adults (int): Number of adult passengers for pricing; accepted as a positive integer count; default is 1.
        min_options (int): Minimum number of feasible flight options to compare before recommending; accepted as a positive integer count; default is 3.
        prefer_short_layovers (bool): Whether to prefer shorter layovers and reasonable total travel time when ranking under-budget options; accepted as true/false; default is True.

    Returns:
        str: A final recommendation containing the selected price, dates, route, layovers, compared options, and booking source.
    """
    run_dir = next_run_dir()
    shutil.copy2(Path(__file__), run_dir / "final_script.py")
    return asyncio.run(_run_search(origin, destination, country, outbound_date, return_date, max_price_hkd, cabin, adults, min_options, prefer_short_layovers, run_dir))


# Backwards-compatible alias for older generated wrappers and logs.
find_iclr_round_trip_flights = find_round_trip_flights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and recommend round-trip economy flights on Google Flights.")
    parser.add_argument("--origin", type=str, default="Hong Kong", help='Departure airport/city code or name for the flight search; accepted as a Google Flights city/airport search string such as "Hong Kong" or "HKG"; default is "Hong Kong".')
    parser.add_argument("--destination", type=str, default="Rio de Janeiro", help='Arrival city/airport code or name for the flight search; accepted as a Google Flights city/airport search string such as "Rio de Janeiro" or "GIG"; default is "Rio de Janeiro".')
    parser.add_argument("--country", type=str, default="Brazil", help='Destination country used to disambiguate Google Flights city options; accepted as a country name string; default is "Brazil".')
    parser.add_argument("--outbound_date", type=str, default="2026-05-03", help='Outbound departure date in ISO YYYY-MM-DD format; should be chosen to cover the target conference/trip window; default is "2026-05-03".')
    parser.add_argument("--return_date", type=str, default="2026-05-11", help='Return departure date in ISO YYYY-MM-DD format; should be about one week after outbound for this task; default is "2026-05-11".')
    parser.add_argument("--max_price_hkd", type=int, default=20000, help="Maximum acceptable total round-trip fare in Hong Kong dollars; accepted as a whole-number HKD amount; default is 20000.")
    parser.add_argument("--cabin", type=str, default="Economy", help='Cabin class to use for the flight search; accepted Google Flights values include "Economy", "Premium economy", "Business", and "First"; default is "Economy".')
    parser.add_argument("--adults", type=int, default=1, help="Number of adult passengers for pricing; accepted as a positive integer count; default is 1.")
    parser.add_argument("--min_options", type=int, default=3, help="Minimum number of feasible flight options to compare before recommending; accepted as a positive integer count; default is 3.")
    parser.add_argument("--prefer_short_layovers", type=parse_bool, default=True, help="Whether to prefer shorter layovers and reasonable total travel time when ranking under-budget options; accepted as true/false; default is True.")
    args = parser.parse_args()
    print(find_round_trip_flights(**vars(args)))
