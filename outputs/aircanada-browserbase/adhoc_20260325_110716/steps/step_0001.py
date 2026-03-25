await page.goto('https://www.aircanada.com/us/en/aco/home.html', wait_until='domcontentloaded')
await page.wait_for_load_state('networkidle')