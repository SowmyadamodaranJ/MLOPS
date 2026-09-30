"""
E2E test configuration for Playwright-based browser tests.

Provides a `page` fixture backed by a Chromium browser instance.
The fixture is scoped per-function so each test gets a clean page.
"""
import pytest


@pytest.fixture(scope="session")
def browser_context():
    """Launch a headless Chromium instance for the E2E test session."""
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1280, "height": 720})
    yield context
    context.close()
    browser.close()
    pw.stop()


@pytest.fixture
def page(browser_context):
    """Yield a fresh Playwright page for each test function."""
    pg = browser_context.new_page()
    yield pg
    pg.close()
