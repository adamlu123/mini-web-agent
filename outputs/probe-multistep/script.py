

# Step 1
await page.get_by_role('button', name='Increment').click()


# Step 2
value = await page.locator('#value').text_content()
assert value == '1', value


# Step 3
await page.get_by_role('button', name='Increment').click()
value = await page.locator('#value').text_content()
assert value == '2', value
