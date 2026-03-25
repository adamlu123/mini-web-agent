await page.get_by_role('textbox', name='Depart Date').click()
for _ in range(3):
    await page.locator('a,button').filter(has=page.locator('text=›')).first.click()
    await asyncio.sleep(0.5)
await page.get_by_text('9', exact=True).nth(1).click()
await page.get_by_text('4', exact=True).last.click()
await page.get_by_role('button', name='Done').click()
await page.locator('#submitbtn').click()