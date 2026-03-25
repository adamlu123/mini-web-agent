booking = page.locator("section, div").filter(has_text=re.compile(r"Hi, where would you like to go\?", re.I)).first

from_box = page.get_by_role("textbox", name="From")
await from_box.click()
await from_box.fill("Singapore")
await asyncio.sleep(1.2)
from_option = page.get_by_role("option").filter(has_text=re.compile(r"Singapore.*SIN|SIN.*Singapore", re.I)).first
if await from_option.count():
    await from_option.click()
else:
    await page.keyboard.press("ArrowDown")
    await page.keyboard.press("Enter")

to_box = page.get_by_role("textbox", name="To")
await to_box.click()
await to_box.fill("Tokyo")
await asyncio.sleep(1.2)
to_option = page.get_by_role("option").filter(has_text=re.compile(r"Tokyo.*(NRT|HND)|(?:NRT|HND).*(Tokyo)", re.I)).first
if await to_option.count():
    await to_option.click()
else:
    await page.keyboard.press("ArrowDown")
    await page.keyboard.press("Enter")