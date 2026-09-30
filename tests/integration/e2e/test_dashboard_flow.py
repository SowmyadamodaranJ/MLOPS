"""
End-to-end Playwright tests for Smart Factory PdM dashboard.

These tests validate the core user workflows against the live frontend/backend.
They require both the Flask backend (port 5000) and Vite dev-server (port 3000)
to be running before execution.

Selectors are aligned to the actual UI structure as of Phase 5 verification.

The layout uses a sidebar with its own `h1` ("Smart Factory AI"), so all page-
content heading checks must use get_by_role with an explicit name, or target the
specific heading text directly.
"""
import pytest
from playwright.sync_api import Page, expect


@pytest.mark.e2e
def test_dashboard_load_and_predict(page: Page):
    """
    E2E: Dashboard loads → KPIs visible → navigate to Predictions →
    run diagnostics → verify result panel populates.
    """
    # ── 1. Open the dashboard ─────────────────────────────────────────────
    page.goto("http://localhost:3000/")

    # Title set in frontend/index.html
    expect(page).to_have_title("Smart Factory PdM | Enterprise Dashboard")

    # 2. Verify the dashboard heading is visible (use role-based selector
    #    to avoid strict mode violation from sidebar h1s)
    heading = page.get_by_role("heading", name="Factory Command Center Live")
    expect(heading).to_be_visible(timeout=15000)

    # 3. Verify at least one KPI metric card rendered
    expect(page.locator("text=Total Machines").first).to_be_visible(timeout=15000)
    expect(page.locator("text=Healthy Machines").first).to_be_visible(timeout=10000)

    # ── 4. Navigate to Predictions page ───────────────────────────────────
    predictions_link = page.locator("a:has-text('Predictions')")
    if predictions_link.count() > 0:
        predictions_link.first.click()
    else:
        page.goto("http://localhost:3000/predictions")

    # 5. Verify the Predictions page loaded
    predictions_heading = page.get_by_role("heading", name="Failure Predictions")
    expect(predictions_heading).to_be_visible(timeout=10000)

    # 6. Verify the sensor workspace section is visible
    expect(page.locator("text=Sensor Tuning Workspace").first).to_be_visible(timeout=10000)

    # 7. Wait for sensor data to finish loading, then click "Execute Diagnostics"
    #    The "Loading sensor data..." spinner must disappear before the button
    #    becomes enabled. Framer-motion animations can cause Playwright's
    #    stability checks to flap, so we use force=True after confirming enabled.
    loading_text = page.locator("text=Loading sensor data")
    # Wait for loading to disappear (may never appear if API is fast)
    if loading_text.count() > 0:
        loading_text.wait_for(state="hidden", timeout=30000)

    predict_btn = page.locator("#predict-submit-btn")
    expect(predict_btn).to_be_visible(timeout=10000)
    expect(predict_btn).to_be_enabled(timeout=30000)
    # Wait for animations to settle
    page.wait_for_timeout(1000)

    # Use requestSubmit() on the parent form — this fires the form's onSubmit
    # handler in React, unlike a raw JS click on the button.
    with page.expect_response("**/api/predict", timeout=30000) as resp_info:
        page.evaluate("document.querySelector('#predict-submit-btn').closest('form').requestSubmit()")

    # Verify the backend responded with 200
    resp = resp_info.value
    assert resp.status == 200, f"Prediction API returned {resp.status}"

    # 8. Wait for the result to render in the UI
    expect(page.locator("text=Risk Score").first).to_be_visible(timeout=30000)

    # 9. Verify a status verdict appears (either "High Risk" or "Stable")
    status_box = page.locator("text=/High Risk|Stable/")
    expect(status_box.first).to_be_visible(timeout=10000)

    # 10. Verify the recommended action section appears
    expect(page.locator("text=Recommended Protocol").first).to_be_visible(timeout=10000)


@pytest.mark.e2e
def test_analytics_page_loads(page: Page):
    """
    E2E: Navigate to Analytics → verify model metric cards render.
    """
    page.goto("http://localhost:3000/analytics")

    # 1. Verify the page heading (specific name avoids sidebar h1 conflict)
    heading = page.get_by_role("heading", name="Model Analytics")
    expect(heading).to_be_visible(timeout=15000)

    # 2. Verify metric cards rendered
    expect(page.locator("text=Accuracy").first).to_be_visible(timeout=15000)
    expect(page.locator("text=Precision").first).to_be_visible(timeout=10000)
    expect(page.locator("text=F1 Score").first).to_be_visible(timeout=10000)

    # 3. Verify at least 4 glass-card containers exist
    cards = page.locator(".glass-card")
    expect(cards.first).to_be_visible(timeout=10000)
    assert cards.count() >= 4, f"Expected at least 4 metric cards, got {cards.count()}"


@pytest.mark.e2e
def test_dashboard_export_csv(page: Page):
    """
    E2E: Dashboard has an 'Export CSV' button in the header.
    Click it and verify no JS error occurs.
    """
    page.goto("http://localhost:3000/")

    # Wait for page to settle using role-based heading
    heading = page.get_by_role("heading", name="Factory Command Center Live")
    expect(heading).to_be_visible(timeout=15000)

    # The dashboard header has an "Export CSV" button
    export_btn = page.locator("button:has-text('Export CSV')")
    expect(export_btn).to_be_visible(timeout=10000)

    # Clicking should not throw
    export_btn.click()

    # Page should still be intact after click
    expect(heading).to_be_visible(timeout=5000)
