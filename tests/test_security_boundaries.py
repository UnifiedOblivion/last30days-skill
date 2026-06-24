"""Regression tests for agent-host local-read boundaries."""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import last30days as cli


def test_importing_cli_does_not_load_config_or_propagate_endpoints(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)
    with mock.patch("lib.env.get_config", side_effect=AssertionError("import loaded config")):
        importlib.reload(cli)
    assert os.environ.get("OPENAI_BASE_URL") is None
    assert os.environ.get("XAI_BASE_URL") is None


def test_diagnose_uses_plan_only_cookie_policy_and_safe_pipeline(monkeypatch):
    seen: dict[str, object] = {}

    def fake_get_config(*, policy):
        seen["policy"] = policy
        return {"_BROWSER_COOKIE_MODE": policy.browser_cookies, "_BROWSER_COOKIE_BROWSERS": ["firefox"]}

    with mock.patch.object(cli.env, "get_config", side_effect=fake_get_config), \
         mock.patch.object(cli.pipeline, "diagnose", return_value={"ok": True}) as diagnose, \
         mock.patch.object(sys, "argv", ["last30days.py", "--diagnose"]):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            assert cli.main() == 0

    assert seen["policy"].browser_cookies == "plan_only"
    diagnose.assert_called_once_with(
        {"_BROWSER_COOKIE_MODE": "plan_only", "_BROWSER_COOKIE_BROWSERS": ["firefox"]},
        None,
        safe=True,
    )
    assert json.loads(stdout.getvalue()) == {"ok": True}


def test_setup_without_cookie_flag_disables_browser_cookie_setup(monkeypatch):
    with mock.patch.object(cli.env, "get_config", return_value={}), \
         mock.patch("lib.setup_wizard.run_auto_setup", return_value={"cookies_found": {}}) as setup, \
         mock.patch("lib.setup_wizard.write_setup_config", return_value=True), \
         mock.patch("lib.setup_wizard.get_setup_status_text", return_value="ok"), \
         mock.patch.object(sys, "argv", ["last30days.py", "setup"]):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            assert cli.main() == 0

    assert setup.call_args.kwargs["allow_browser_cookies"] is False


def test_setup_cookie_flag_allows_browser_cookie_setup(monkeypatch):
    with mock.patch.object(cli.env, "get_config", return_value={}), \
         mock.patch("lib.setup_wizard.run_auto_setup", return_value={"cookies_found": {}}) as setup, \
         mock.patch("lib.setup_wizard.write_setup_config", return_value=True), \
         mock.patch("lib.setup_wizard.get_setup_status_text", return_value="ok"), \
         mock.patch.object(sys, "argv", ["last30days.py", "setup", "--allow-browser-cookies"]):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            assert cli.main() == 0

    assert setup.call_args.kwargs["allow_browser_cookies"] is True


def test_no_browser_cookies_overrides_setup_cookie_flag(monkeypatch):
    seen: dict[str, object] = {}

    def fake_get_config(*, policy):
        seen["policy"] = policy
        return {}

    with mock.patch.object(cli.env, "get_config", side_effect=fake_get_config), \
         mock.patch("lib.setup_wizard.run_auto_setup", return_value={"cookies_found": {}}) as setup, \
         mock.patch("lib.setup_wizard.write_setup_config", return_value=True), \
         mock.patch("lib.setup_wizard.get_setup_status_text", return_value="ok"), \
         mock.patch.object(
             sys,
             "argv",
             ["last30days.py", "--no-browser-cookies", "setup", "--allow-browser-cookies"],
         ):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            assert cli.main() == 0

    assert seen["policy"].browser_cookies == "off"
    assert setup.call_args.kwargs["allow_browser_cookies"] is False


def test_diagnose_overrides_setup_cookie_flag(monkeypatch):
    seen: dict[str, object] = {}

    def fake_get_config(*, policy):
        seen["policy"] = policy
        return {}

    with mock.patch.object(cli.env, "get_config", side_effect=fake_get_config), \
         mock.patch("lib.setup_wizard.run_auto_setup", return_value={"cookies_found": {}}) as setup, \
         mock.patch("lib.setup_wizard.write_setup_config", return_value=True), \
         mock.patch("lib.setup_wizard.get_setup_status_text", return_value="ok"), \
         mock.patch.object(
             sys,
             "argv",
             ["last30days.py", "--diagnose", "setup", "--allow-browser-cookies"],
         ):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            assert cli.main() == 0

    assert seen["policy"].browser_cookies == "plan_only"
    assert setup.call_args.kwargs["allow_browser_cookies"] is False
