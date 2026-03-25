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