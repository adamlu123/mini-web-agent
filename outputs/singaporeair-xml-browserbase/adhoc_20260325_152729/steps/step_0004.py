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