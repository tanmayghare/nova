import asyncio

from nova.browser import Browser, BrowserConfig


async def main() -> None:
    """Run the browser example."""
    # Create a browser instance with custom configuration
    config = BrowserConfig(
        headless=False,  # Show the browser window
        viewport={"width": 1280, "height": 800},
    )
    browser = Browser(config)

    try:
        # Start the browser
        await browser.start()

        # Navigate to a website
        await browser.navigate("https://example.com")

        # Get the page title
        title = await browser.get_text("h1")
        print(f"Page title: {title}")

        # Take a screenshot
        await browser.screenshot("example.png")
        print("Screenshot saved as example.png")

    finally:
        # Always stop the browser
        await browser.stop()


if __name__ == "__main__":
    asyncio.run(main())
