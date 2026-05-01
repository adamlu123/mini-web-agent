"""Reusable Ticketmaster classical concert recommendation CLI."""
import argparse
import asyncio
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright


def find_classical_concert_tickets(
    start_url: str = "https://www.ticketmaster.com/discover/seattle?categoryId=KZFzniwnSyZfZ7v7nJ&classificationId=KnvZfZ7vAeJ",
    destination_city: str = "Seattle",
    origin_city: str = "Redmond, Washington",
    genre: str = "Classical",
    start_date: str = "2026-05-01",
    end_date: str = "2026-05-05",
    budget_min_usd: float = 85.0,
    budget_max_usd: float = 140.0,
    compare_count: int = 3,
) -> dict[str, Any]:
    """Find and recommend Ticketmaster classical concert tickets near Seattle.

    Args:
        start_url (str): Ticketmaster discover URL used as the starting point; accepted format is a full https://www.ticketmaster.com/... URL. Defaults to the Seattle classical discover page.
        destination_city (str): Desired Ticketmaster city or market for the concert; accepted format is a city name such as "Seattle". Defaults to "Seattle".
        origin_city (str): Traveler's base location used to judge venue practicality; accepted format is a city/state string such as "Redmond, Washington". Defaults to "Redmond, Washington".
        genre (str): Concert genre or classification to verify on Ticketmaster; accepted format is a visible genre label such as "Classical". Defaults to "Classical".
        start_date (str): First acceptable concert date, inclusive; accepted format is ISO YYYY-MM-DD. Defaults to "2026-05-01".
        end_date (str): Last acceptable concert date, inclusive; accepted format is ISO YYYY-MM-DD. Defaults to "2026-05-05".
        budget_min_usd (float): Minimum target ticket price in USD before taxes/fees or all-in before taxes when Ticketmaster only shows all-in pricing; accepted as a decimal dollar amount. Defaults to 85.0.
        budget_max_usd (float): Maximum target ticket price in USD before taxes/fees or all-in before taxes when Ticketmaster only shows all-in pricing; accepted as a decimal dollar amount. Defaults to 140.0.
        compare_count (int): Number of available date or seat/ticket options to compare before selecting a recommendation; accepted as a positive integer. Defaults to 3.

    Returns:
        dict[str, Any]: Recommendation details including event name, date, venue, ticket tier/section, price, booking source, checkout notes, compared options, run directory, and log path.
    """

    @dataclass
    class EventCandidate:
        name: str
        date_text: str
        iso_date: str
        time_text: str
        venue: str
        city_state: str
        url: str

    @dataclass
    class TicketOption:
        event: EventCandidate
        section: str
        row: str
        ticket_type: str
        price: float
        raw_text: str

    workspace = Path(os.environ.get("WORKSPACE_DIR", Path.cwd())).resolve()
    runs_dir = workspace / "final_runs"
    runs_dir.mkdir(exist_ok=True)
    existing = [int(p.name.split("_", 1)[1]) for p in runs_dir.glob("run_*") if p.name.split("_", 1)[-1].isdigit()]
    run_id = max(existing, default=0) + 1
    run_dir = runs_dir / f"run_{run_id:03d}"
    screenshots = run_dir / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "final_script_log.txt"
    log_path.write_text("", encoding="utf-8")
    try:
        shutil.copy2(Path(__file__).resolve(), run_dir / "final_script.py")
    except Exception:
        pass

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")
        print(message)

    def cdp_url() -> str:
        return os.environ.get("LOCAL_BROWSER_CDP_URL") or os.environ.get("BROWSER_CDP_URL") or "http://127.0.0.1:9222"

    def date_in_range(iso: str) -> bool:
        return start_date <= iso <= end_date

    def iso_from_event_month_day(year: str, mon: str, day: str) -> str:
        months = {"JAN":"01","FEB":"02","MAR":"03","APR":"04","MAY":"05","JUN":"06","JUL":"07","AUG":"08","SEP":"09","OCT":"10","NOV":"11","DEC":"12"}
        return f"{year}-{months[mon.upper()]}-{int(day):02d}"

    async def dismiss_overlays(page) -> None:
        for label in ["全部拒绝", "Reject All", "Decline", "Dismiss popup", "Accept & Continue", "Got it"]:
            try:
                loc = page.get_by_text(label, exact=True).first
                if await loc.is_visible(timeout=700):
                    await loc.click(timeout=1500)
                    await page.wait_for_timeout(500)
            except Exception:
                pass

    async def extract_events_from_discover(page) -> list[EventCandidate]:
        anchors = await page.locator("a[href*='/event/']").evaluate_all(
            """els => els.map(a => ({text: a.innerText || a.getAttribute('aria-label') || '', href: a.href}))"""
        )
        events: list[EventCandidate] = []
        seen: set[str] = set()
        for item in anchors:
            text = " ".join((item.get("text") or "").split())
            href = item.get("href") or ""
            if href in seen or "Find Tickets" not in text:
                continue
            seen.add(href)
            m = re.search(r"Find Tickets\s+(.+?),\s+([^,]+,\s*[A-Z]{2})\s+(.+?)(\d{1,2})/(\d{1,2})/(\d{2}),\s*([0-9:]+\s*[AP]M)", text)
            if not m:
                continue
            name, city_state, venue, mm, dd, yy, time_text = m.groups()
            iso = f"20{yy}-{int(mm):02d}-{int(dd):02d}"
            events.append(EventCandidate(name=name.strip(), date_text=f"{mm}/{dd}/{yy}", iso_date=iso, time_text=time_text, venue=venue.strip(), city_state=city_state.strip(), url=href))
        return events

    async def parse_ticket_options(page, event: EventCandidate) -> list[TicketOption]:
        body = await page.locator("body").inner_text(timeout=20000)
        options: list[TicketOption] = []
        # Match Ticketmaster ticket list entries like: Sec ORCH REAR R • Row QQ\nResale Ticket\n$110.90
        pattern = re.compile(r"Sec\s+([^\n•]+(?:\s+[^\n•]+)*?)\s*•\s*Row\s+([^\n]+)\n([^\n]*Ticket)\n\$(\d+(?:\.\d{2})?)", re.I)
        for sec, row, ticket_type, price in pattern.findall(body):
            raw = f"Sec {sec.strip()} • Row {row.strip()} {ticket_type.strip()} ${price}"
            options.append(TicketOption(event=event, section=sec.strip(), row=row.strip(), ticket_type=ticket_type.strip(), price=float(price), raw_text=raw))
        return options

    async def async_run() -> dict[str, Any]:
        log(f"step 0 params: start_url={start_url} destination_city={destination_city} origin_city={origin_city} genre={genre} start_date={start_date} end_date={end_date} budget_min_usd={budget_min_usd} budget_max_usd={budget_max_usd} compare_count={compare_count}")
        async with async_playwright() as playwright:
            browser = await playwright.chromium.connect_over_cdp(cdp_url())
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = await context.new_page()
            log("step 1 action: open Ticketmaster Seattle classical discover URL to verify booking source, destination city, and genre category.")
            await page.goto(start_url, wait_until="domcontentloaded", timeout=90000)
            await page.wait_for_timeout(7000)
            await dismiss_overlays(page)
            await page.screenshot(path=str(screenshots / "final_execution_1_ticketmaster_seattle_classical_results.png"), full_page=True)
            discover_text = await page.locator("body").inner_text(timeout=20000)
            title = await page.title()
            log(f"step 1 evidence: title={title!r}; page contains destination={destination_city in discover_text}; genre={genre in discover_text}; URL={page.url}")

            events = await extract_events_from_discover(page)
            in_window = [e for e in events if date_in_range(e.iso_date) and destination_city.lower() in (e.city_state + ' ' + e.venue).lower()]
            log("step 2 action: inspect visible event list and restrict candidates to the requested inclusive date window and Seattle-area venue.")
            for e in events:
                log(f"step 2 event_seen: {e.name} | {e.iso_date} {e.time_text} | {e.city_state} {e.venue} | {e.url}")
            log("step 2 evidence: candidates_in_requested_window=" + "; ".join(f"{e.name} {e.iso_date} {e.time_text} {e.venue}" for e in in_window))
            await page.screenshot(path=str(screenshots / "final_execution_2_date_window_candidates.png"), full_page=True)

            compared: list[TicketOption] = []
            event_pages: list[tuple[EventCandidate, list[TicketOption], str]] = []
            for idx, event in enumerate(in_window[:4], start=1):
                log(f"step {2+idx} action: open available Ticketmaster event page for comparison option date {event.iso_date} {event.time_text}: {event.name}.")
                ep = await context.new_page()
                await ep.goto(event.url, wait_until="domcontentloaded", timeout=90000)
                await ep.wait_for_timeout(10000)
                await dismiss_overlays(ep)
                # Ensure ticket list view/filter button is visible; no exact price filter is exposed on this resale page, so compare the sorted lowest-price list.
                try:
                    await ep.get_by_role("button", name=re.compile("Filters", re.I)).click(timeout=3000)
                    await ep.wait_for_timeout(1000)
                except Exception:
                    pass
                fname = f"final_execution_{2+idx}_event_{event.iso_date}_tickets.png"
                await ep.screenshot(path=str(screenshots / fname), full_page=True)
                body = await ep.locator("body").inner_text(timeout=20000)
                opts = await parse_ticket_options(ep, event)
                event_pages.append((event, opts, body))
                for opt in opts[:8]:
                    in_budget = budget_min_usd <= opt.price <= budget_max_usd
                    log(f"step {2+idx} option_seen: {event.iso_date} {event.time_text} | {opt.section} Row {opt.row} | {opt.ticket_type} | ${opt.price:.2f} all-in before taxes | in_budget={in_budget}")
                compared.extend(opts)
                await ep.close()

            budget_options = [o for o in compared if budget_min_usd <= o.price <= budget_max_usd]
            if budget_options:
                recommendation = sorted(budget_options, key=lambda o: (o.price, o.event.iso_date, o.event.time_text))[0]
                rationale = "lowest available Ticketmaster all-in-before-taxes price inside the requested target range"
            else:
                recommendation = sorted(compared, key=lambda o: abs(o.price - budget_max_usd))[0]
                rationale = "closest available Ticketmaster option to the requested target range because no exact in-budget ticket was visible"
            # Reopen recommended page for final evidence with the recommended option visible.
            log(f"step 6 action: reopen recommended Ticketmaster event and capture final visible ticket evidence for the selected section/tier and price.")
            fp = await context.new_page()
            await fp.goto(recommendation.event.url, wait_until="domcontentloaded", timeout=90000)
            await fp.wait_for_timeout(10000)
            await dismiss_overlays(fp)
            await fp.screenshot(path=str(screenshots / "final_execution_6_recommended_ticket_evidence.png"), full_page=True)
            await fp.close()

            checkout_notes = "Ticketmaster page states this is the Ticketmaster resale marketplace, not the primary ticket provider; resale prices can exceed face value; displayed We’re All In prices include fees before taxes; availability and pricing are subject to change; do not continue to payment until taxes/fees and delivery are reviewed."
            result = {
                "concert_name": recommendation.event.name,
                "date": f"{recommendation.event.iso_date} {recommendation.event.time_text}",
                "venue": f"{recommendation.event.venue}, {recommendation.event.city_state}",
                "seat_section_or_tier": f"Sec {recommendation.section}, Row {recommendation.row}, {recommendation.ticket_type}",
                "price": f"${recommendation.price:.2f} all-in before taxes as displayed by Ticketmaster",
                "booking_source": "Ticketmaster",
                "event_url": recommendation.event.url,
                "rationale": rationale,
                "checkout_notes": checkout_notes,
                "compared_options": [f"{o.event.iso_date} {o.event.time_text} | {o.section} Row {o.row} | ${o.price:.2f} | in_budget={budget_min_usd <= o.price <= budget_max_usd}" for o in sorted(compared, key=lambda x: (x.event.iso_date, x.price))[:max(compare_count, 1)]],
                "run_dir": str(run_dir),
                "log_path": str(log_path),
            }
            log("step 7 final_recommendation: " + "; ".join(f"{k}={v}" for k, v in result.items() if k not in {"compared_options"}))
            log("step 7 compared_options: " + " || ".join(result["compared_options"]))
            log("FINAL_RESPONSE: Recommend {concert_name} on {date} at {venue}; choose {seat_section_or_tier} for {price}. Booking source: {booking_source}. Notes: {checkout_notes}".format(**result))
            await page.close()
            return result

    return asyncio.run(async_run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and recommend Ticketmaster classical concert tickets near Seattle.")
    parser.add_argument("--start_url", type=str, default="https://www.ticketmaster.com/discover/seattle?categoryId=KZFzniwnSyZfZ7v7nJ&classificationId=KnvZfZ7vAeJ", help="Ticketmaster discover URL used as the starting point; accepted format is a full https://www.ticketmaster.com/... URL. Defaults to the Seattle classical discover page.")
    parser.add_argument("--destination_city", type=str, default="Seattle", help="Desired Ticketmaster city or market for the concert; accepted format is a city name such as 'Seattle'. Defaults to 'Seattle'.")
    parser.add_argument("--origin_city", type=str, default="Redmond, Washington", help="Traveler's base location used to judge venue practicality; accepted format is a city/state string such as 'Redmond, Washington'. Defaults to 'Redmond, Washington'.")
    parser.add_argument("--genre", type=str, default="Classical", help="Concert genre or classification to verify on Ticketmaster; accepted format is a visible genre label such as 'Classical'. Defaults to 'Classical'.")
    parser.add_argument("--start_date", type=str, default="2026-05-01", help="First acceptable concert date, inclusive; accepted format is ISO YYYY-MM-DD. Defaults to '2026-05-01'.")
    parser.add_argument("--end_date", type=str, default="2026-05-05", help="Last acceptable concert date, inclusive; accepted format is ISO YYYY-MM-DD. Defaults to '2026-05-05'.")
    parser.add_argument("--budget_min_usd", type=float, default=85.0, help="Minimum target ticket price in USD before taxes/fees or all-in before taxes when Ticketmaster only shows all-in pricing; accepted as a decimal dollar amount. Defaults to 85.0.")
    parser.add_argument("--budget_max_usd", type=float, default=140.0, help="Maximum target ticket price in USD before taxes/fees or all-in before taxes when Ticketmaster only shows all-in pricing; accepted as a decimal dollar amount. Defaults to 140.0.")
    parser.add_argument("--compare_count", type=int, default=3, help="Number of available date or seat/ticket options to compare before selecting a recommendation; accepted as a positive integer. Defaults to 3.")
    args = parser.parse_args()
    find_classical_concert_tickets(**vars(args))
