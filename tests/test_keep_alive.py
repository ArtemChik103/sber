from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from scripts.keep_alive import DEFAULT_SLEEP_SELECTORS, parse_args, wake_up_app


def test_parse_args_defaults() -> None:
    with patch.object(sys, "argv", ["keep_alive.py"]):
        with patch.dict("os.environ", {"STREAMLIT_APP_URL": "https://test.streamlit.app"}):
            args = parse_args()
            assert args.url == "https://test.streamlit.app"
            assert args.timeout_sec == 60


def test_wake_up_empty_url_fails() -> None:
    exit_code = wake_up_app("")
    assert exit_code == 1


def test_wake_up_clicks_button_when_sleeping() -> None:
    fake_browser = MagicMock()
    fake_context = MagicMock()
    fake_page = MagicMock()

    fake_browser.new_context.return_value = fake_context
    fake_context.new_page.return_value = fake_page

    fake_page.content.return_value = "<html>This app has gone to sleep due to inactivity</html>"
    fake_button = MagicMock()
    fake_button.is_visible.return_value = True
    fake_page.locator.return_value.first = fake_button

    with patch("scripts.keep_alive.sync_playwright") as mock_playwright:
        mock_p_instance = MagicMock()
        mock_p_instance.chromium.launch.return_value = fake_browser
        mock_playwright.return_value.__enter__.return_value = mock_p_instance

        exit_code = wake_up_app("https://my-app.streamlit.app")

        assert exit_code == 0
        fake_button.click.assert_called_once()
        fake_browser.close.assert_called_once()


def test_wake_up_succeeds_when_already_active() -> None:
    fake_browser = MagicMock()
    fake_context = MagicMock()
    fake_page = MagicMock()

    fake_browser.new_context.return_value = fake_context
    fake_context.new_page.return_value = fake_page

    fake_page.content.return_value = "<html>App loaded and active</html>"
    fake_button = MagicMock()
    fake_button.is_visible.return_value = False
    fake_page.locator.return_value.first = fake_button

    fake_app_view = MagicMock()
    fake_app_view.is_visible.return_value = True

    def locator_side_effect(selector: str):
        mock_loc = MagicMock()
        if 'stAppViewContainer' in selector or '.stApp' in selector:
            mock_loc.first = fake_app_view
        else:
            mock_loc.first = fake_button
        return mock_loc

    fake_page.locator.side_effect = locator_side_effect

    with patch("scripts.keep_alive.sync_playwright") as mock_playwright:
        mock_p_instance = MagicMock()
        mock_p_instance.chromium.launch.return_value = fake_browser
        mock_playwright.return_value.__enter__.return_value = mock_p_instance

        exit_code = wake_up_app("https://my-app.streamlit.app")

        assert exit_code == 0
        fake_button.click.assert_not_called()
        fake_browser.close.assert_called_once()
