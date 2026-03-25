

# Step 1
body_text = await page.locator("body").inner_text()
for name in ["Accept", "I accept", "Agree", "Allow all", "OK", "Continue", "Got it"]:
    btn = page.get_by_role("button", name=name)
    if await btn.count():
        try:
            await btn.first.click(timeout=2000)
            break
        except:
            pass

await page.screenshot(path="step1.png", full_page=True)


# Step 2
booking = page.locator("section, div").filter(has_text=re.compile(r"Hi, where would you like to go\?", re.I)).first

from_box = page.get_by_role("textbox", name="From")
await from_box.click()
await from_box.fill("Singapore")
await asyncio.sleep(1.2)
from_option = page.get_by_role("option").filter(has_text=re.compile(r"Singapore.*SIN|SIN.*Singapore", re.I)).first
if await from_option.count():
    await from_option.click()
else:
    await page.keyboard.press("ArrowDown")
    await page.keyboard.press("Enter")

to_box = page.get_by_role("textbox", name="To")
await to_box.click()
await to_box.fill("Tokyo")
await asyncio.sleep(1.2)
to_option = page.get_by_role("option").filter(has_text=re.compile(r"Tokyo.*(NRT|HND)|(?:NRT|HND).*(Tokyo)", re.I)).first
if await to_option.count():
    await to_option.click()
else:
    await page.keyboard.press("ArrowDown")
    await page.keyboard.press("Enter")


# Step 3
target_month = "June 2026"

for _ in range(6):
    snapshot = await page.locator("body").aria_snapshot()
    if target_month in snapshot:
        break
    await page.locator('a[href="#"]').last.click()
    await asyncio.sleep(0.5)

snapshot = await page.locator("body").aria_snapshot()
if target_month not in snapshot:
    raise RuntimeError("Could not navigate date picker to June 2026")

await page.get_by_text("June 2026", exact=True).locator("..").get_by_text("9", exact=True).click()
await asyncio.sleep(0.5)


# Step 4
await page.locator("text=June 2026").wait_for(state="visible", timeout=10000)

june_header = page.get_by_text("June 2026", exact=True).first
header_box = await june_header.bounding_box()
if header_box is None:
    raise RuntimeError("June 2026 header not visible in date picker")

day_nines = page.get_by_text("9", exact=True)
target = None
best_dx = float("inf")

for i in range(await day_nines.count()):
    cand = day_nines.nth(i)
    box = await cand.bounding_box()
    if box is None:
        continue
    if box["x"] <= header_box["x"]:
        continue
    dx = abs((box["x"] + box["width"] / 2) - (header_box["x"] + header_box["width"] / 2))
    if dx < best_dx:
        best_dx = dx
        target = cand

if target is None:
    raise RuntimeError("Could not find June 9 day cell in visible date picker")

await target.click()
await asyncio.sleep(0.8)


# Step 5
overlay = page.locator("body").locator("text=May 2026").locator("..").locator("..")
await overlay.wait_for(state="visible")

box = await overlay.bounding_box()
if box is None:
    raise RuntimeError("Date picker overlay bounding box not available")

mid_x = box["x"] + box["width"] / 2

items = overlay.get_by_role("listitem")
target = None
for i in range(await items.count()):
    item = items.nth(i)
    text = (await item.inner_text()).strip()
    if not text:
        continue
    normalized = re.sub(r"\s+", " ", text)
    if not normalized.startswith("9"):
        continue
    ibox = await item.bounding_box()
    if ibox is None:
        continue
    center_x = ibox["x"] + ibox["width"] / 2
    if center_x > mid_x:
        target = item
        break

if target is None:
    raise RuntimeError("Could not find clickable June 9 cell in right-hand calendar")

await target.click()
await asyncio.sleep(0.8)


# Step 6
await page.mouse.click(794, 248)
await asyncio.sleep(1)


# Step 7
await page.get_by_role("textbox", name="Depart Date").click()


# Step 8
next_month = page.locator('a[href="#"]').filter(has=page.locator("svg")).last
for _ in range(2):
    await next_month.click()
    await asyncio.sleep(0.8)

await page.mouse.click(795, 249)
await asyncio.sleep(1.2)
