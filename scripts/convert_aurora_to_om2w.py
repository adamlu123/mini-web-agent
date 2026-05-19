#!/usr/bin/env python3
"""Convert aurora_segment_heldout_test_sets_v2 task_data.json files to
om2w_260220.json-like format: [{task_id, confirmed_task, website, task_type}, ...].
"""
import json
import os
import re
import sys

ROOT = "/home/luyadong/sandbox/aurora_segment_heldout_test_sets_v2_20260415_precomputed_rubric"
OUT = "/home/luyadong/sandbox/mini-web-agent/src/miniswewebagent/run/benchmarks/aurora_heldout_v2_260415.json"

# Folder-prefix -> canonical host. Extend as needed; fallback = <prefix>.com.
BRAND_HOST = {
    # airlines
    "aerlingus": "aerlingus.com",
    "airasia": "airasia.com",
    "aircanada": "aircanada.com",
    "alaskaair": "alaskaair.com",
    "alitalia": "ita-airways.com",
    "allegiantair": "allegiantair.com",
    "ana": "ana.co.jp",
    "britishairways": "britishairways.com",
    "cathaypacific": "cathaypacific.com",
    "delta": "delta.com",
    "easyjet": "easyjet.com",
    "emirates": "emirates.com",
    "etihad": "etihad.com",
    "frontier": "flyfrontier.com",
    "hawaiianair": "hawaiianairlines.com",
    "iberia": "iberia.com",
    "jetblue": "jetblue.com",
    "klm": "klm.com",
    "lufthansa": "lufthansa.com",
    "qantas": "qantas.com",
    "qatarairways": "qatarairways.com",
    "ryanair": "ryanair.com",
    "singaporeair": "singaporeair.com",
    "southwest": "southwest.com",
    "spirit": "spirit.com",
    "thaiairways": "thaiairways.com",
    "turkishairlines": "turkishairlines.com",
    "united": "united.com",
    "vueling": "vueling.com",
    "wizzair": "wizzair.com",
    # hotels
    "agoda": "agoda.com",
    "airbnb": "airbnb.com",
    "booking": "booking.com",
    "hilton": "hilton.com",
    "hyatt": "hyatt.com",
    "marriott": "marriott.com",
    "ihg": "ihg.com",
    "accor": "all.accor.com",
    "choicehotels": "choicehotels.com",
    "wyndham": "wyndhamhotels.com",
    "bestwestern": "bestwestern.com",
    "radisson": "radissonhotels.com",
    "expedia": "expedia.com",
    "priceline": "priceline.com",
    "kayak": "kayak.com",
    "trivago": "trivago.com",
    "hotels": "hotels.com",
    # shopping
    "amazon": "amazon.com",
    "walmart": "walmart.com",
    "target": "target.com",
    "bestbuy": "bestbuy.com",
    "costco": "costco.com",
    "samsclub": "samsclub.com",
    "ebay": "ebay.com",
    "etsy": "etsy.com",
    "shein": "shein.com",
    "wayfair": "wayfair.com",
    "potterybarn": "potterybarn.com",
    "macys": "macys.com",
    "nordstrom": "nordstrom.com",
    "jcpenney": "jcpenney.com",
    "kohls": "kohls.com",
    "homedepot": "homedepot.com",
    "lowes": "lowes.com",
    "ikea": "ikea.com",
    "zappos": "zappos.com",
    "aliexpress": "aliexpress.com",
    "fiestafactorydirect": "fiestafactorydirect.com",
    "acrylux": "acrylux.com",
    "agwheelexpress": "agwheelexpress.com",
    "americanstandard-us": "americanstandard-us.com",
    # activities / trips
    "alltrails": "alltrails.com",
    "tripadvisor": "tripadvisor.com",
    "viator": "viator.com",
    "getyourguide": "getyourguide.com",
    "klook": "klook.com",
    "disneyworld": "disneyworld.disney.go.com",
    # ticketing
    "ticketmaster": "ticketmaster.com",
    "stubhub": "stubhub.com",
    "seatgeek": "seatgeek.com",
    "vividseats": "vividseats.com",
    "eventbrite": "eventbrite.com",
    # food
    "yelp": "yelp.com",
    "opentable": "opentable.com",
    "doordash": "doordash.com",
    "ubereats": "ubereats.com",
    "grubhub": "grubhub.com",
    # real estate
    "zillow": "zillow.com",
    "redfin": "redfin.com",
    "realtor": "realtor.com",
    "trulia": "trulia.com",
    "compass": "compass.com",
    "remax": "remax.com",
    # jobs
    "indeed": "indeed.com",
    "linkedin": "linkedin.com",
    "glassdoor": "glassdoor.com",
    "ziprecruiter": "ziprecruiter.com",
    "monster": "monster.com",
}

DOMAIN_RE = re.compile(
    r"\b([a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+\.(?:com|org|net|co|edu|gov|io|info|us|uk|de|fr|jp|in|ca|au))\b",
    re.I,
)
SIMPLE_DOMAIN_RE = re.compile(
    r"\b([a-z0-9][a-z0-9-]*\.(?:com|org|net|co|edu|gov|io|info))\b", re.I
)
# Skip generic tokens when deriving brand from folder name
GENERIC_TOKENS = {
    "apply", "salary", "range", "buy", "rent", "book", "find", "other",
    "music", "event", "comparison", "shopping", "lottery", "plan", "trip",
    "a", "the", "of", "in", "for", "to", "eat", "dine", "at", "house",
    "condo", "home", "apartment", "citation", "composite", "search",
    "add", "cart", "list", "new", "used", "question", "answering",
    "website", "spoton",
    # jobs-specific generic sub-categories
    "employer", "job", "jobs", "benefits", "qualifications", "responsibilities",
    "wording", "wildcard", "pay", "requirements", "description", "title",
    "position", "role",
}
# Task types where folder prefix is never a brand (e.g., restaurant names,
# real-estate city names). Always use the category default.
NON_BRAND_CATEGORIES = {
    "restaurants_tail",
    "realestate_complex",
    "jobs",
}
# If the first (non-generic) token of the folder name falls in this set,
# the folder prefix is not a real brand — fall back to a category default.
NON_BRAND_FIRST_TOKENS = {
    "composite", "apply", "salary", "buy", "rent", "find",
}
# Per task_type fallback when no brand/domain can be inferred.
CATEGORY_DEFAULT_HOST = {
    "compositional_tasks_v2": "google.com",
    "flights": "google.com/travel/flights",
    "hotels_head": "booking.com",
    "jobs": "indeed.com",
    "price_comparison": "google.com/shopping",
    "realestate_complex": "zillow.com",
    "restaurants_tail": "yelp.com",
    "shopping_head": "amazon.com",
    "shopping_lists_tail": "amazon.com",
    "things_to_do": "tripadvisor.com",
    "ticketing": "ticketmaster.com",
}
# Two-letter US state codes we shouldn't pick as a "brand"
STATE_CODES = {
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id","il","in",
    "ia","ks","ky","la","me","md","ma","mi","mn","ms","mo","mt","ne","nv",
    "nh","nj","nm","ny","nc","nd","oh","ok","or","pa","ri","sc","sd","tn",
    "tx","ut","vt","va","wa","wv","wi","wy",
}

def website_from_proposal(proposal: str):
    if not proposal:
        return None
    # Multi-dot first (e.g., us.megabus.com), then simple
    for regex in (DOMAIN_RE, SIMPLE_DOMAIN_RE):
        for m in regex.finditer(proposal):
            host = m.group(1).lower().rstrip(".")
            # filter obvious non-website mentions
            if host.endswith((".us",)) and len(host.split(".")) == 2:
                continue
            return host
    return None

def brand_from_folder(folder: str):
    # strip trailing _<digits>
    name = re.sub(r"_\d+$", "", folder)
    tokens = [t for t in name.split("_") if t]
    for t in tokens:
        tl = t.lower()
        if tl in GENERIC_TOKENS:
            continue
        if tl in STATE_CODES:
            continue
        if tl.isdigit():
            continue
        if len(tl) <= 1:
            continue
        return tl
    return tokens[0].lower() if tokens else None

def host_to_url(host: str) -> str:
    host = host.lower().rstrip("/")
    if not host.startswith("www.") and host.count(".") == 1:
        host = "www." + host
    return f"https://{host}/"

def infer_website(folder: str, proposal: str, task_type: str):
    # 1) explicit domain mentioned in proposal
    host = website_from_proposal(proposal)
    if host:
        return host_to_url(host)
    # 2) map folder-prefix brand to known host
    brand = brand_from_folder(folder)
    if brand and brand in BRAND_HOST:
        return host_to_url(BRAND_HOST[brand])
    # 3) categories where the folder prefix is never a real brand
    #    (restaurant names, cities, job sub-types) -> category default.
    if task_type in NON_BRAND_CATEGORIES:
        default = CATEGORY_DEFAULT_HOST.get(task_type)
        if default:
            return host_to_url(default)
    # 4) if first token of folder is generic (composite_*, apply_*, buy_*, ...),
    #    fall back to the category default rather than guessing <token>.com.
    first_token = folder.split("_")[0].lower()
    if first_token in NON_BRAND_FIRST_TOKENS:
        default = CATEGORY_DEFAULT_HOST.get(task_type)
        if default:
            return host_to_url(default)
    # 5) fallback: guess <brand>.com
    if brand:
        return host_to_url(f"{brand}.com")
    # 6) last-resort category default
    default = CATEGORY_DEFAULT_HOST.get(task_type)
    if default:
        return host_to_url(default)
    return ""

def main():
    out = []
    for cat in sorted(os.listdir(ROOT)):
        cat_dir = os.path.join(ROOT, cat)
        if not os.path.isdir(cat_dir):
            continue
        for folder in sorted(os.listdir(cat_dir)):
            td = os.path.join(cat_dir, folder, "task_data.json")
            if not os.path.isfile(td):
                continue
            try:
                d = json.load(open(td))
            except Exception as e:
                print(f"skip {td}: {e}", file=sys.stderr)
                continue
            proposal = (d.get("task_proposal")
                        or d.get("task_summary") or "").strip()
            task_type = d.get("task_type") or cat
            init_url = (d.get("init_url") or "").strip()
            website = init_url or infer_website(folder, proposal, task_type)
            out.append({
                "task_id": folder,
                "confirmed_task": proposal,
                "website": website,
                "task_type": task_type,
            })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)
    print(f"wrote {len(out)} tasks -> {OUT}")

if __name__ == "__main__":
    main()
