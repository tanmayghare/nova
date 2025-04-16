# scraper.py
import os
from playwright.sync_api import sync_playwright, Page
from pathlib import Path
import time
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
TARGET_SITES = {
    "expedia": "https://www.expedia.com/",
    "booking": "https://www.booking.com/"
}

SCENARIOS = {
    "expedia": [
        {"name": "homepage", "url": TARGET_SITES["expedia"]},
        # {"name": "flight_search_sample", "url": "https://www.expedia.com/Flights-Search?flight-type=on&mode=search&trip=roundtrip&leg1=from:London,to:New York,departure:10/15/2024&leg2=from:New York,to:London,departure:10/22/2024&options=cabinclass:economy&passengers=adults:1"},
    ],
    "booking": [
        {"name": "homepage", "url": TARGET_SITES["booking"]},
    ]
}


OUTPUT_DIR = Path("data/raw")
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
DOM_DIR = OUTPUT_DIR / "dom"

# Increased delays
DELAY_BETWEEN_SITES = 5  # Seconds between processing different websites
DELAY_BETWEEN_SCENARIOS = 8 # Seconds between scenarios for the *same* site
DELAY_AFTER_NAVIGATE = 5 # Initial wait after page.goto before manual check
DELAY_BEFORE_CAPTURE = 3 # Wait just before taking screenshot/DOM

# --- Helper Functions ---

def safe_filename(name: str) -> str:
    """Creates a safe filename from a scenario name or URL."""
    name = re.sub(r'[:/\\?*"<>|]', '_', name) # Remove invalid chars
    return name[:100] # Limit length

def ensure_dir(path: Path):
    """Ensures a directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def wait_for_manual_intervention(page: Page, site_name: str, scenario_name: str):
    """Pauses script execution for manual actions like CAPTCHA solving."""
    logging.warning(f"--- PAUSING for manual intervention on {site_name} - {scenario_name} ---")
    logging.warning("Please check the browser window. Solve any CAPTCHAs or perform necessary actions.")
    input(">>> Press Enter in this terminal when ready to continue...")
    logging.info("--- Resuming script ---")
    # Add a small delay after resuming to allow any post-CAPTCHA scripts to run
    time.sleep(2)

def capture_page(page: Page, site_name: str, scenario_name: str):
    """Captures screenshot and DOM, saving them to the respective directories."""
    try:
        # Wait for page to load reasonably well (adjust time as needed)
        # Using 'load' or 'domcontentloaded' might be sufficient if 'networkidle' causes issues
        page.wait_for_load_state('load', timeout=20000)
        logging.info(f"Waiting {DELAY_BEFORE_CAPTURE}s before capture...")
        time.sleep(DELAY_BEFORE_CAPTURE) # Wait for final rendering adjustments

        screenshot_path = SCREENSHOT_DIR / site_name / f"{safe_filename(scenario_name)}.png"
        dom_path = DOM_DIR / site_name / f"{safe_filename(scenario_name)}.html"

        ensure_dir(screenshot_path.parent)
        ensure_dir(dom_path.parent)

        logging.info(f"Capturing screenshot: {screenshot_path}")
        page.screenshot(path=screenshot_path, full_page=True)

        logging.info(f"Saving DOM: {dom_path}")
        html_content = page.content()
        with open(dom_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    except Exception as e:
        logging.error(f"Error capturing {site_name} - {scenario_name}: {e}")


# --- Main Scraping Logic ---

def main():
    ensure_dir(SCREENSHOT_DIR)
    ensure_dir(DOM_DIR)

    with sync_playwright() as p:
        # Run with browser window visible
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        # Set a realistic user agent
        page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
        # Set viewport size
        page.set_viewport_size({"width": 1920, "height": 1080})


        for site_name, scenarios in SCENARIOS.items():
            logging.info(f"--- Processing site: {site_name} ---")
            for i, scenario in enumerate(scenarios):
                scenario_name = scenario["name"]
                logging.info(f"Processing scenario: {scenario_name}")

                if "url" in scenario:
                    try:
                        logging.info(f"Navigating to {scenario['url']}...")
                        # Increased navigation timeout
                        page.goto(scenario["url"], timeout=60000, wait_until='domcontentloaded')
                        logging.info(f"Initial navigation complete. Waiting {DELAY_AFTER_NAVIGATE}s...")
                        time.sleep(DELAY_AFTER_NAVIGATE)

                        # <<< MANUAL INTERVENTION POINT >>>
                        wait_for_manual_intervention(page, site_name, scenario_name)

                        # Proceed with capture after manual step
                        capture_page(page, site_name, scenario_name)

                    except Exception as e:
                        logging.error(f"Failed during navigation or capture for {scenario['url']}: {e}")

                elif "actions" in scenario:
                    # Placeholder for action-based scenarios
                    logging.warning(f"Scenario '{scenario_name}' action execution not implemented yet.")
                    # You would likely need a wait_for_manual_intervention call here too,
                    # potentially after performing the actions but before the final capture.
                    pass

                # Delay between scenarios of the same site
                if i < len(scenarios) - 1:
                     logging.info(f"Waiting {DELAY_BETWEEN_SCENARIOS}s before next scenario...")
                     time.sleep(DELAY_BETWEEN_SCENARIOS)

            # Delay between different sites
            logging.info(f"Waiting {DELAY_BETWEEN_SITES}s before next site...")
            time.sleep(DELAY_BETWEEN_SITES)


        logging.info("--- Scraping potentially finished, closing browser ---")
        # Add a final pause before closing if needed for review
        input(">>> Scraping loop complete. Press Enter to close the browser...")
        browser.close()
        logging.info("--- Browser closed ---")

if __name__ == "__main__":
    # Remember: 'playwright install' might be needed first
    main()