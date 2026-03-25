await page.get_by_role('button', name='Increment').click()
value = await page.locator('#value').text_content()
assert value == '2', value