try:
    from playwright.sync_api import sync_playwright
    print("Playwright is installed and working correctly")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        print("Successfully launched Chromium browser")
        browser.close()
except Exception as e:
    print(f"Error with Playwright: {e}") 