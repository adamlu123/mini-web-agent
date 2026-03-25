value = await page.locator('#value').text_content()
assert value == '1', value