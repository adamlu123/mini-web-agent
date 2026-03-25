to_box = page.get_by_role('textbox', name='To')
await to_box.click()
await to_box.fill('Tokyo')
await page.wait_for_timeout(1000)