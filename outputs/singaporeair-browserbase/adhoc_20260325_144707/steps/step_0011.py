await page.evaluate("window.scrollTo({top: 0, behavior: 'auto'})")
await page.wait_for_timeout(1000)