import json
import threading
import urllib.request
from pathlib import Path

import pytest

from leanroute import Leanroute
from leanroute.cli import main
from leanroute.dashboard import PAGE, make_server
from leanroute.usage import LocalUsage

from test_sdk import FakeEngine

REPO_PAGE = Path(__file__).resolve().parents[2] / "web" / "dashboard.html"


def test_usage_is_saved_with_prices(tmp_path, monkeypatch):
    db = str(tmp_path / "u.db")
    monkeypatch.setenv("LEANROUTE_DB", db)
    lr = Leanroute(engine=FakeEngine(difficulty=0.4), policy=None, project="shop",
                   prices={"small": (0.1, 0.4), "big": (1.0, 4.0)})
    d = lr.route("hi", cheap="small", strong="big")
    lr.stats.record(d, "big", 1000, 500)
    r = LocalUsage(db).usage_report(days=7)
    t = r["totals"]
    assert r["source"] == "local" and t["requests"] == 1 and t["priced_requests"] == 1
    assert t["routed_cheap"] == 1 and t["saved_usd"] == pytest.approx(0.0027)
    assert r["pricing"]["cheap_model"] == "small" and r["daily"][0]["requests"] == 1
    assert LocalUsage(db).usage_report(project="other")["totals"]["requests"] == 0


def test_usage_without_prices_is_counted_but_unpriced(tmp_path):
    u = LocalUsage(str(tmp_path / "u.db"))
    u.record(project="default", route="strong", model="big", cost_usd=None, baseline_usd=None, decision_ms=5)
    t = u.usage_report()["totals"]
    assert t["requests"] == 1 and t["priced_requests"] == 0 and t["saved_usd"] == 0


def test_recording_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setenv("LEANROUTE_RECORD", "0")
    assert Leanroute(engine=FakeEngine()).stats.store is None
    monkeypatch.delenv("LEANROUTE_RECORD")
    assert Leanroute(engine=FakeEngine(), record_usage=False).stats.store is None


def test_dashboard_server_serves_page_and_usage(tmp_path):
    db = str(tmp_path / "u.db")
    LocalUsage(db).record(project="p", route="cheap", model="small", cost_usd=0.001, baseline_usd=0.01, decision_ms=3)
    server = make_server(port=0, db=db)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        html = urllib.request.urlopen(base + "/dashboard").read()
        assert b"<html" in html.lower()
        data = json.loads(urllib.request.urlopen(base + "/v1/usage?days=7").read())
        assert data["source"] == "local" and data["totals"]["saved_usd"] == pytest.approx(0.009)
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/nope")
    finally:
        server.shutdown()
        server.server_close()


def test_cli_stats(tmp_path, capsys):
    db = str(tmp_path / "u.db")
    LocalUsage(db).record(project="p", route="blocked", model=None, cost_usd=0.0, baseline_usd=0.0, decision_ms=3)
    assert main(["stats", "--db", db]) == 0
    assert json.loads(capsys.readouterr().out)["blocked"] == 1


@pytest.mark.skipif(not REPO_PAGE.exists(), reason="not in the repo checkout")
def test_bundled_page_matches_the_server_dashboard():
    assert PAGE.read_bytes() == REPO_PAGE.read_bytes(), "copy web/dashboard.html to sdk/src/leanroute/"
