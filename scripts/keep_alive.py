from __future__ import annotations

import argparse
import os
import sys
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


DEFAULT_SLEEP_SELECTORS = [
    'button:has-text("Yes, get this app back up!")',
    'button:has-text("get this app back up")',
    'button:has-text("back up")',
    'button:has-text("Wake up")',
    'button:has-text("Yes")',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep-alive ping and wake-up utility for Streamlit Cloud.")
    parser.add_argument(
        "--url",
        default=os.environ.get("STREAMLIT_APP_URL", ""),
        help="Target URL of the deployed Streamlit application.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=60,
        help="Page load timeout in seconds.",
    )
    return parser.parse_args()


def wake_up_app(url: str, timeout_sec: int = 60) -> int:
    clean_url = url.strip()
    if not clean_url:
        print("ERROR: Streamlit app URL is empty. Provide --url or set STREAMLIT_APP_URL environment variable.", file=sys.stderr)
        return 1

    print(f"Starting keep-alive check for: {clean_url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()

        try:
            print(f"Navigating to {clean_url}...")
            page.goto(clean_url, timeout=timeout_sec * 1000, wait_until="domcontentloaded")
            page.wait_for_timeout(5000)
        except PlaywrightTimeoutError:
            print("WARNING: Navigation timed out waiting for DOMContentLoaded. Checking current page content...")

        page_content = page.content()
        is_sleeping = "has gone to sleep" in page_content or "sleeping" in page_content.lower()

        wake_button = None
        for sel in DEFAULT_SLEEP_SELECTORS:
            try:
                locator = page.locator(sel).first
                if locator.is_visible(timeout=2000):
                    wake_button = locator
                    print(f"Found sleep wake-up button matching selector: {sel}")
                    break
            except Exception:
                continue

        if wake_button is not None:
            print("Action: App is sleeping. Clicking wake-up button...")
            wake_button.click()
            print("Waiting 15 seconds for app container restoration...")
            page.wait_for_timeout(15000)
            print("Success: Wake-up signal dispatched to Streamlit Cloud.")
            browser.close()
            return 0

        if is_sleeping:
            print("WARNING: Detected sleep text, but wake button was not interactable. Refreshing page...")
            page.reload()
            page.wait_for_timeout(8000)

        # Check if the Streamlit app container rendered
        try:
            app_view = page.locator('[data-testid="stAppViewContainer"], .stApp').first
            if app_view.is_visible(timeout=5000):
                print("Success: Streamlit app is active and responsive.")
                browser.close()
                return 0
        except Exception:
            pass

        print(f"Status: Page loaded. Current page title: '{page.title()}'.")
        browser.close()
        return 0


def main() -> None:
    args = parse_args()
    exit_code = wake_up_app(args.url, timeout_sec=args.timeout_sec)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
