to_box = page.get_by_role('textbox', name='To')
await to_box.click()
await page.get_by_text('Tokyo, Japan Tokyo (ALL)', exact=False).click()