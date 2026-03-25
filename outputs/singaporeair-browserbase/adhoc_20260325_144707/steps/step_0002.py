await page.wait_for_load_state('networkidle')
await page.wait_for_timeout(3000)