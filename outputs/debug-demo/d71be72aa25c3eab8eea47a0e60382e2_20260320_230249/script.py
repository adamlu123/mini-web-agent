

# Step 1
await page.get_by_role('link', name=re.compile('Tech Specs|Specifications', re.I)).first.click()
