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