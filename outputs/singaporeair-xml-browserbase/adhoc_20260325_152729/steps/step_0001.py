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