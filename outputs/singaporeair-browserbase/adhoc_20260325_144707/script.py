

# Step 1
await page.goto('https://www.singaporeair.com/en_UK/us/home', wait_until='domcontentloaded')


# Step 2
await page.wait_for_load_state('networkidle')
await page.wait_for_timeout(3000)


# Step 3
await page.get_by_role('button', name='Accept').click()


# Step 4
from_box = page.get_by_role('textbox', name='From')
await from_box.click()
await from_box.press('Meta+A')
await from_box.fill('Singapore')
await page.wait_for_timeout(1000)


# Step 5
to_box = page.get_by_role('textbox', name='To')
await to_box.click()
await to_box.fill('Tokyo')
await page.wait_for_timeout(1000)


# Step 6
to_box = page.get_by_role('textbox', name='To')
await to_box.click()
await page.get_by_text('Tokyo, Japan Tokyo (ALL)', exact=False).click()


# Step 7
await page.get_by_role('link', name='').last.click()


# Step 8
await page.locator('textbox[name="Depart Date"], input[aria-label="Depart Date"]').first().scroll_into_view_if_needed()
await page.locator('textbox[name="Depart Date"], input[aria-label="Depart Date"]').first().click()


# Step 9
await page.locator('text=Hi, where would you like to go?').scroll_into_view_if_needed()
await asyncio.sleep(1)


# Step 10
await page.goto('https://www.singaporeair.com/en_UK/us/home#/book/bookflight')


# Step 11
await page.evaluate("window.scrollTo({top: 0, behavior: 'auto'})")
await page.wait_for_timeout(1000)


# Step 12
await page.get_by_role('textbox', name='Depart Date').click()
await asyncio.sleep(1)


# Step 13
await page.locator('a[href="#"]').last.click()


# Step 14
await page.locator('div[role="dialog"], .calendar-root, .ui-datepicker, body').get_by_role('link').nth(1).click()


# Step 15
await page.goto('https://www.singaporeair.com/en_UK/us/home#/book/bookflight')


# Step 16
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


# Step 17
await page.evaluate("""() => {
  const setVal = (el, val) => {
    if (!el) return;
    el.focus();
    const native = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    if (native) native.call(el, val); else el.value = val;
    el.dispatchEvent(new Event('input', {bubbles:true}));
    el.dispatchEvent(new Event('change', {bubbles:true}));
    el.dispatchEvent(new Event('blur', {bubbles:true}));
  };
  const inputs = Array.from(document.querySelectorAll('input'));
  const depart = inputs.find(i => (i.getAttribute('aria-label')||'').includes('Depart Date') || (i.getAttribute('placeholder')||'').includes('Depart'));
  const ret = inputs.find(i => (i.getAttribute('aria-label')||'').includes('Return Date') || (i.getAttribute('placeholder')||'').includes('Return'));
  setVal(depart, '09/06/2026');
  setVal(ret, '04/07/2026');
}""")
await asyncio.sleep(1)
await page.get_by_role("button", name="Search", exact=True).click()
try:
    await page.wait_for_url(lambda url: 'flightsearch' in url or 'bookflight' not in url, timeout=20000)
except:
    pass


# Step 18
await page.get_by_role('textbox', name='Depart Date').click()
for _ in range(3):
    await page.locator('a,button').filter(has=page.locator('text=›')).first.click()
    await asyncio.sleep(0.5)
await page.get_by_text('9', exact=True).nth(1).click()
await page.get_by_text('4', exact=True).last.click()
await page.get_by_role('button', name='Done').click()
await page.locator('#submitbtn').click()
