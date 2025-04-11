import asyncio

from playwright.async_api import async_playwright


async def install_browsers():
    """Install Playwright browser binaries."""
    print("Installing Playwright browser binaries...")
    playwright = await async_playwright().start()
    await playwright.chromium.launch()
    await playwright.firefox.launch()
    await playwright.webkit.launch()
    await playwright.stop()
    print("Installation complete!")


if __name__ == "__main__":
    asyncio.run(install_browsers())
