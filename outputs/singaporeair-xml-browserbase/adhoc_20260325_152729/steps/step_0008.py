next_month = page.locator('a[href="#"]').filter(has=page.locator("svg")).last
for _ in range(2):
    await next_month.click()
    await asyncio.sleep(0.8)

await page.mouse.click(795, 249)
await asyncio.sleep(1.2)