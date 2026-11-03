"""Deployment modes deny before they spend: no visitor outside local mode reaches the provider unless the owner allowed guests, the preview
answers nothing but its invite page without a valid cookie, and the public mode has no Syrus session or turn route at all. Offline only:
a counting model double that fails the test if it is ever called."""
import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from backend.access import COOKIE, Gate, resolve
from backend.syrus_app.llm import Completion
from backend.syrus_app.settings import load_settings
from backend.web import create_web

CODE = "judge-panel-2026"
HASH = hashlib.sha256(CODE.encode()).hexdigest()


class Counting:
    def __init__(self):
        self.calls = 0

    async def complete(self, messages, schema, max_tokens, timeout):
        self.calls += 1
        return Completion("not json", 100, 10)


def settings(tmp_path):
    return load_settings({"SYRUS_LLM_API_KEY": "test-not-a-key", "SYRUS_CAPS_CONFIRMED": "1", "SYRUS_BUDGET_FILE": str(tmp_path / "budget.json")})


def web(tmp_path, env, adapter=None):
    return create_web(settings(tmp_path), adapter, dist=tmp_path / "none", access=resolve(env))


def case(c):
    return c.get("/api/cases").json()["cases"][0]["case_id"]


def enter(c, code=CODE):
    return c.post("/preview/enter", content=f"code={code}", headers={"Content-Type": "application/x-www-form-urlencoded"}, follow_redirects=False)


# --- configuration refuses unsafe setups -------------------------------------------------------------------------------------

@pytest.mark.parametrize("env", [{"AEROXAI_MODE": "demo"}, {"AEROXAI_MODE": "preview"}, {"AEROXAI_MODE": "preview", "AEROXAI_PREVIEW_CODE_SHA256": "abc"},
                                 {"AEROXAI_MODE": "public", "SYRUS_DIAGNOSTICS": "1"}])
def test_unsafe_configurations_stop_the_server(env):
    with pytest.raises(RuntimeError):
        resolve(env)


def test_the_old_public_flag_still_means_public():
    assert resolve({"AEROXAI_PUBLIC": "1"}).mode == "public" and resolve({}).mode == "local"


# --- public ------------------------------------------------------------------------------------------------------------------

def test_public_has_no_syrus_session_or_turn_route_and_no_simulation(tmp_path):
    fake = Counting()
    c = TestClient(web(tmp_path, {"AEROXAI_MODE": "public"}, fake))
    assert c.get("/api/cases").status_code == 200 and c.get(f"/api/cases/{case(c)}?mode=retro").status_code == 200
    assert c.get("/api/health").json()["reason"] == "guests_off"
    assert c.post("/api/session", json={"case_id": case(c), "mode": "retro"}).status_code in (404, 405)
    assert c.post("/api/syrus/turn", json={"message": "Why?"}, headers={"Authorization": "Bearer x"}).status_code in (404, 405)
    for path in ("/lab/twin", "/twin/simulate", "/optimization/recommend"):
        assert c.post(path, json={}).status_code in (404, 405), path
    assert c.get("/config").json() == {"mode": "public", "features": {"syrus_chat": False, "lab": "off", "twin": False, "optimization": False}}
    assert fake.calls == 0 and c.app.state.svc.adapter is None and c.app.state.svc.settings.key is None


# --- preview, guests off -----------------------------------------------------------------------------------------------------

PREVIEW = {"AEROXAI_MODE": "preview", "AEROXAI_PREVIEW_CODE_SHA256": HASH}


def test_the_preview_shows_only_its_invite_page_without_a_cookie(tmp_path):
    c = TestClient(web(tmp_path, PREVIEW))
    page = c.get("/")
    assert page.status_code == 200 and "Invite code" in page.text and "<script" not in page.text
    for path in ("/api/cases", "/explorer/overview", "/config", "/evidence/summary"):
        r = c.get(path)
        assert r.status_code == 401 and r.json()["error"]["code"] == "preview_locked", path
    assert c.post("/api/session", json={"case_id": "c-h3vz", "mode": "retro"}).status_code == 401
    assert c.get("/health").status_code == 200


def test_a_wrong_code_is_refused_and_guessing_is_limited(tmp_path):
    c = TestClient(web(tmp_path, PREVIEW))
    assert enter(c, "wrong").status_code == 401 and COOKIE not in c.cookies
    for _ in range(9):
        enter(c, "wrong")
    assert enter(c, CODE).status_code == 429  # the eleventh try in ten minutes is refused even with the right code


def test_the_right_code_opens_the_preview_with_a_signed_http_only_cookie(tmp_path):
    c = TestClient(web(tmp_path, PREVIEW))
    r = enter(c)
    assert r.status_code == 303 and r.headers["location"] == "/"
    head = r.headers["set-cookie"].lower()
    assert "httponly" in head and "samesite=strict" in head
    assert c.get("/api/cases").status_code == 200
    assert c.get("/config").json() == {"mode": "preview", "features": {"syrus_chat": False, "lab": "live", "twin": False, "optimization": False}}
    assert c.post("/twin/simulate", json={}).status_code in (404, 405) and c.post("/lab/twin", json={}).status_code == 200


def test_a_tampered_or_expired_cookie_is_refused():
    gate = Gate(resolve(PREVIEW), now=lambda: 1000.0)
    good = gate.cookie()
    vid, expires, sig = good.split(".")
    assert gate.visitor(good) == vid
    assert gate.visitor(f"{vid}.{int(expires) + 999}.{sig}") is None and gate.visitor(f"other.{expires}.{sig}") is None and gate.visitor("junk") is None
    later = Gate(gate.a, now=lambda: float(expires) + 1)
    assert later.visitor(good) is None


def test_with_guests_off_no_turn_reaches_the_provider(tmp_path):
    fake = Counting()
    c = TestClient(web(tmp_path, PREVIEW, fake))
    enter(c)
    assert c.get("/api/health").json() == {**c.get("/api/health").json(), "status": "offline", "reason": "guests_off"}
    s = c.post("/api/session", json={"case_id": case(c), "mode": "retro"}).json()
    assert s["chat_available"] is False and s["offline_reason"] == "guests_off"
    r = c.post("/api/syrus/turn", json={"message": "Why did this alert fire?"}, headers={"Authorization": f"Bearer {s['token']}"})
    assert r.json()["reason"] == "guests_off" and fake.calls == 0 and c.app.state.svc.adapter is None


# --- preview, guests live ----------------------------------------------------------------------------------------------------

def test_live_guests_get_the_lower_daily_cap_and_a_turn_quota(tmp_path):
    fake = Counting()
    c = TestClient(web(tmp_path, {**PREVIEW, "AEROXAI_SYRUS_GUESTS": "live", "AEROXAI_GUEST_TURNS": "2", "AEROXAI_PREVIEW_DAILY_USD": "1.5"}, fake))
    assert c.app.state.svc.budget.caps["daily"] == 1.5 and c.app.state.svc.budget.caps["total"] == 30.0
    enter(c)
    assert c.get("/config").json()["features"]["syrus_chat"] is True
    s = c.post("/api/session", json={"case_id": case(c), "mode": "retro"}).json()
    auth = {"Authorization": f"Bearer {s['token']}"}
    codes = []
    for q in ("Why did this alert fire?", "What evidence is missing?", "Was this definitely a leak?"):
        r = c.post("/api/syrus/turn", json={"message": q}, headers=auth)
        codes.append(r.json().get("error", {}).get("code"))
    assert codes[2] == "guest_quota"
    calls = fake.calls
    assert c.post("/api/syrus/turn", json={"message": "One more?"}, headers=auth).status_code == 429 and fake.calls == calls


def test_per_visitor_request_limits_apply(tmp_path):
    env = {**PREVIEW}
    a = resolve(env)
    gate = Gate(type(a)(**{**a.__dict__, "requests_per_minute": 3, "lab_runs_per_10_minutes": 1}), now=lambda: 50.0)
    assert [gate.allow("v", "/explorer/overview", "GET") for _ in range(4)] == [None, None, None, "rate_limited"]
    other = Gate(type(a)(**{**a.__dict__, "lab_runs_per_10_minutes": 1}))
    assert other.allow("v", "/lab/twin", "POST") is None and other.allow("v", "/lab/twin", "POST") == "lab_rate_limited"


# --- local is unchanged ------------------------------------------------------------------------------------------------------

def test_local_serves_everything_and_keeps_the_owner_caps(tmp_path):
    c = TestClient(web(tmp_path, {}, Counting()))
    assert c.get("/api/cases").status_code == 200 and c.get("/config").json()["features"] == {"syrus_chat": True, "lab": "live", "twin": True, "optimization": True}
    assert c.app.state.svc.budget.caps["daily"] == 10.0 and c.app.state.svc.adapter is not None


# --- the launcher ------------------------------------------------------------------------------------------------------------

def test_the_launcher_binds_locally_with_one_worker_and_trusts_only_local_proxy_headers(monkeypatch, tmp_path):
    import scripts.run_preview as run

    seen = {}
    monkeypatch.setattr(run.uvicorn, "run", lambda app, **kw: seen.update(kw, app=app))
    for k, v in {**PREVIEW, "SYRUS_BUDGET_FILE": str(tmp_path / "b.json")}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("sys.argv", ["run_preview"])
    run.main()
    assert seen["host"] == "127.0.0.1" and seen["workers"] == 1 and seen["forwarded_allow_ips"] == "127.0.0.1" and seen["app"] == "backend.web:app"


def test_a_second_live_server_refuses_to_share_the_budget(monkeypatch, tmp_path):
    import scripts.run_preview as run

    monkeypatch.setattr(run.uvicorn, "run", lambda app, **kw: None)
    for k, v in {**PREVIEW, "AEROXAI_SYRUS_GUESTS": "live", "SYRUS_BUDGET_FILE": str(tmp_path / "b.json")}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("sys.argv", ["run_preview"])
    (tmp_path / "b.lock").write_text("123")
    with pytest.raises(SystemExit, match="Another server"):
        run.main()


def test_forwarded_addresses_cannot_be_spoofed_through_a_local_tunnel():
    from uvicorn.middleware.proxy_headers import _TrustedHosts

    # cloudflared on 127.0.0.1 appends the real address; a client-supplied value stays on the left and is ignored
    assert _TrustedHosts("127.0.0.1").get_trusted_client_address("6.6.6.6, 203.0.113.7")[0] == "203.0.113.7"


def test_nothing_secret_is_in_the_config_answer(tmp_path):
    c = TestClient(web(tmp_path, {}, Counting()))
    assert "key" not in json.dumps(c.get("/config").json()).lower()
