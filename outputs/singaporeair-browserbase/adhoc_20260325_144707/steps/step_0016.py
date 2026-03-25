from playwright.async_api import expect

# Fill origin
from_box = page.get_by_role('textbox', name='From')
await from_box.click()
await from_box.fill('Singapore')
await page.get_by_text('Singapore, Singapore', exact=False).first.click()

# Fill destination
to_box = page.get_by_role('textbox', name='To')
await to_box.click()
await to_box.fill('Tokyo')
await page.get_by_text('Tokyo, Japan Tokyo (ALL)', exact=False).first.click()

# Open date picker and move to June/July 2026
await page.get_by_role('textbox', name='Depart Date').click()
for _ in range(3):
    await page.locator('a[href="#"]').filter(has=page.locator('text=/^$/')).last.click()
    await page.wait_for_timeout(300)

# Select dates if visible
jun9 = page.get_by_text('9', exact=True)
jul4 = page.get_by_text('4', exact=True)
if await jun9.count() > 0:
    await jun9.first.click()
    await page.wait_for_timeout(300)
if await jul4.count() > 0:
    await jul4.last.click()
    await page.wait_for_timeout(300)

# Close picker if needed and search
if await page.get_by_role('button', name='Done').count() > 0:
    await page.get_by_role('button', name='Done').click()
await page.get_by_role('button', name='Search').click()

# Wait briefly for results page or no-flights message
await page.wait_for_timeout(8000)