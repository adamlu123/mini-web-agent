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