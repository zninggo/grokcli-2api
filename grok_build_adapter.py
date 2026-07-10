"""Adapter: grok-build-auth -> grokcli-2api account pool.

Replaces the legacy email_registration.py flow by driving
``grok-build-auth/xconsole_client`` to:

1. register an x.ai account with temp-mail + YesCaptcha
2. extract SSO/session cookies
3. complete Build OAuth (PKCE + consent) using the signup session
4. import the resulting CLIProxyAPI auth record into grokcli-2api's auth.json

Import of ``xconsole_client`` is deferred so the main API can start even when
optional deps are missing. Registration endpoints then return a clear error
instead of crashing process startup.

``grok-build-auth`` is vendored in-tree (not a git submodule).
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
GBA = ROOT / "grok-build-auth"

YESCAPTCHA_KEY = (
    os.environ.get("GROK2API_YESCAPTCHA_KEY")
    or os.environ.get("YESCAPTCHA_API_KEY")
    or ""
).strip()

# --------------------------------------------------------------------------- #
# session state
# --------------------------------------------------------------------------- #
_sessions: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_xconsole_ready = False
_xconsole_error: str | None = None


def _now() -> float:
    return time.time()


def _clean_old_sessions() -> None:
    cutoff = _now() - 6 * 3600
    for sid in list(_sessions.keys()):
        sess = _sessions.get(sid) or {}
        if float(sess.get("updated_at") or 0) < cutoff:
            _sessions.pop(sid, None)


def _compact_session(sess: dict[str, Any]) -> dict[str, Any]:
    out = dict(sess)
    out.pop("_client", None)
    out.pop("_oauth_client", None)
    if out.get("auth_json"):
        out["auth_json_count"] = len(out["auth_json"])
        out.pop("auth_json", None)
    return out


def ensure_xconsole() -> None:
    """Ensure vendored grok-build-auth/xconsole_client is importable.

    Raises RuntimeError with actionable message when unavailable.
    Safe to call multiple times.
    """
    global _xconsole_ready, _xconsole_error
    if _xconsole_ready:
        return
    if _xconsole_error:
        raise RuntimeError(_xconsole_error)

    if not GBA.is_dir():
        _xconsole_error = (
            "grok-build-auth 目录不存在。请确认仓库完整检出，"
            "或重新 clone 本项目。"
        )
        raise RuntimeError(_xconsole_error)

    xc = GBA / "xconsole_client"
    if not xc.is_dir():
        _xconsole_error = (
            "grok-build-auth/xconsole_client 不存在。"
            "请确认仓库完整检出（该目录已内置，不再使用 git submodule）。"
        )
        raise RuntimeError(_xconsole_error)

    # Put vendored package root on sys.path so `import xconsole_client` works.
    gba_str = str(GBA.resolve())
    if gba_str not in sys.path:
        sys.path.insert(0, gba_str)

    try:
        # Import side-effect: validate package is loadable.
        import xconsole_client  # noqa: F401
        from xconsole_client import (  # noqa: F401
            XConsoleAuthClient,
            YesCaptchaSolver,
            create_solver,
            xai_oauth_login_protocol,
        )
        from xconsole_client.oauth_protocol import (  # noqa: F401
            extract_cookies_from_auth_client,
        )
        from xconsole_client.xai_oauth import (  # noqa: F401
            CLIPROXYAPI_GROK_HEADERS,
            build_cliproxyapi_auth_record,
        )
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        if missing in ("curl_cffi", "requests") or "curl_cffi" in str(e) or "requests" in str(e):
            _xconsole_error = (
                f"注册机依赖缺失: {missing}。请执行: pip install -r requirements.txt"
            )
        else:
            _xconsole_error = (
                f"无法导入 xconsole_client ({e})。请执行: pip install -r requirements.txt"
            )
        raise RuntimeError(_xconsole_error) from e
    except Exception as e:  # noqa: BLE001
        _xconsole_error = f"加载 grok-build-auth 失败: {e}"
        raise RuntimeError(_xconsole_error) from e

    _xconsole_ready = True
    _xconsole_error = None


def registration_available() -> dict[str, Any]:
    """Non-raising health probe for admin UI / startup logs."""
    try:
        ensure_xconsole()
        return {
            "ok": True,
            "available": True,
            "path": str(GBA),
            "vendored": True,
            "yescaptcha_configured": bool(YESCAPTCHA_KEY),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "available": False,
            "path": str(GBA),
            "vendored": True,
            "error": str(e),
            "yescaptcha_configured": bool(YESCAPTCHA_KEY),
        }


# --------------------------------------------------------------------------- #
# mail provider: moemail (reuse grokcli-2api config)
# --------------------------------------------------------------------------- #
def _make_email_receiver(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    prefix: str | None = None,
    domain: str | None = None,
    expiry_ms: int | None = None,
):
    from email_registration import (  # local import: always available in this repo
        _moemail_create_mailbox,
    )
    from config import MOEMAIL_API_KEY, MOEMAIL_BASE_URL, MOEMAIL_DOMAIN, MOEMAIL_EXPIRY_MS

    key = (api_key or MOEMAIL_API_KEY or "").strip()
    if not key:
        raise ValueError(
            "MoeMail API key missing. Set GROK2API_MOEMAIL_API_KEY or pass api_key."
        )
    base = (base_url or MOEMAIL_BASE_URL).rstrip("/")
    dom = (domain or MOEMAIL_DOMAIN).strip(".")
    pre = (prefix or f"grok-{secrets.token_hex(4)}").lower()

    mailbox = _moemail_create_mailbox(
        name=pre,
        domain=dom,
        expiry_ms=expiry_ms if expiry_ms is not None else MOEMAIL_EXPIRY_MS,
        api_key=key,
        base_url=base,
    )
    email_id = mailbox["id"]
    address = mailbox["email"]

    class _MoeMailReceiver:
        def __init__(self, email: str, email_id: str, api_key: str | None, base_url: str | None):
            self.email = email
            self.email_id = email_id
            self.api_key = api_key
            self.base_url = base_url or "https://moemail.521884.xyz"

        def wait_for_code(self, timeout: float = 120) -> str:
            from email_registration import _moemail_fetch_messages

            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    messages = _moemail_fetch_messages(
                        self.email_id,
                        api_key=self.api_key,
                        base_url=self.base_url,
                        include_details=True,
                    )
                    for item in messages:
                        extracted = item.get("extracted") or {}
                        codes = extracted.get("codes") or []
                        for code in codes:
                            clean = str(code).replace("-", "").strip().upper()
                            if len(clean) == 6:
                                return clean
                        text = "\n".join(
                            str(item.get(k) or "")
                            for k in ("subject", "content", "html", "from_address", "from")
                        )
                        match = __import__("re").search(
                            r"\b([A-Z0-9]{3})-([A-Z0-9]{3})\b", text
                        )
                        if match:
                            return "".join(match.groups())
                except Exception:
                    pass
                time.sleep(5)
            raise RuntimeError("timeout waiting for xAI email verification code")

    return address, _MoeMailReceiver(address, email_id, api_key=key, base_url=base)


def _proxy_url() -> str:
    from email_registration import _normalize_proxy_config
    from config import XAI_PROXY

    cfg = _normalize_proxy_config(XAI_PROXY or None)
    return cfg["proxy"] if cfg else ""


# --------------------------------------------------------------------------- #
# registration flow
# --------------------------------------------------------------------------- #
def start_registration(
    *,
    yescaptcha_key: str | None = None,
    proxy: str | None = None,
    moemail_api_key: str | None = None,
    moemail_base_url: str | None = None,
    prefix: str | None = None,
    domain: str | None = None,
    expiry_ms: int | None = None,
) -> dict[str, Any]:
    """Start one registration session and return its public state."""
    try:
        ensure_xconsole()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}

    _clean_old_sessions()

    key = (yescaptcha_key or YESCAPTCHA_KEY or "").strip()
    if not key:
        return {
            "ok": False,
            "error": "YESCAPTCHA_KEY is required (set GROK2API_YESCAPTCHA_KEY or pass yescaptcha_key)",
        }

    try:
        email, receiver = _make_email_receiver(
            api_key=moemail_api_key,
            base_url=moemail_base_url,
            prefix=prefix,
            domain=domain,
            expiry_ms=expiry_ms,
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}

    # xAI password rules: mix upper/lower/digit/symbol.
    password = f"Aa{os.urandom(5).hex()}9!xZ"
    sid = f"gba_{uuid.uuid4().hex[:16]}"

    sess = {
        "id": sid,
        "status": "started",
        "created_at": _now(),
        "updated_at": _now(),
        "email": email,
        "password": password,
        "message": f"started; email={email}",
        "sso": None,
        "oauth": None,
        "auth_json": None,
        "error": None,
        "yescaptcha_key": key,
        "proxy": proxy or _proxy_url(),
    }
    _sessions[sid] = sess

    threading.Thread(
        target=_run_registration,
        args=(sid, key, proxy or _proxy_url(), receiver),
        daemon=True,
        name=f"gba-reg-{sid[-8:]}",
    ).start()

    return {"ok": True, **_compact_session(sess)}


def _run_registration(
    sid: str,
    yescaptcha_key: str,
    proxy: str,
    receiver: Any,
) -> None:
    sess = _sessions.get(sid)
    if not sess:
        return

    def update(status: str, message: str, **kwargs: Any) -> None:
        sess["status"] = status
        sess["message"] = message
        sess["updated_at"] = _now()
        sess.update(kwargs)

    email = str(sess.get("email") or "").strip().lower()
    password = sess["password"]
    sess["email"] = email
    client = None

    try:
        ensure_xconsole()
        from xconsole_client import (
            XConsoleAuthClient,
            YesCaptchaSolver,
            xai_oauth_login_protocol,
        )
        from xconsole_client import config as C
        from xconsole_client.oauth_protocol import extract_cookies_from_auth_client
        from xconsole_client.xai_oauth import (
            CLIPROXYAPI_GROK_HEADERS,
            build_cliproxyapi_auth_record,
        )
        import accounts
        from config import UPSTREAM_BASE

        update("registering", "visiting signup page")
        client = XConsoleAuthClient(
            debug=True,
            proxy=proxy or "",
            signup_url="https://accounts.x.ai/sign-up?redirect=grok-com",
        )
        client.visit_home()
        client.load_signup_page()

        update("registering", "sending email validation code")
        client.create_email_validation_code(email)

        update("waiting_email", "waiting for xAI verification code")
        code = receiver.wait_for_code(timeout=120)
        code = str(code or "").strip().upper().replace(" ", "")
        update("registering", f"code received: {code}")

        client.verify_email_validation_code(email, code)
        client.validate_password(email, password)

        update("solving_turnstile", "solving Turnstile via YesCaptcha")
        sitekey = (
            getattr(client, "turnstile_sitekey", None)
            or getattr(C, "TURNSTILE_SITEKEY", None)
            or ""
        ).strip()
        website_url = (getattr(client, "signup_url", None) or C.SIGNUP_URL or "").strip()
        if not sitekey:
            raise RuntimeError(
                "Turnstile sitekey missing. Signup page scrape failed and "
                "config TURNSTILE_SITEKEY is empty."
            )

        endpoint = (
            os.environ.get("GROK2API_YESCAPTCHA_ENDPOINT")
            or os.environ.get("YESCAPTCHA_ENDPOINT")
            or ""
        ).strip() or None

        def _turnstile_progress(msg: str) -> None:
            update("solving_turnstile", f"Turnstile: {msg}")

        solver = YesCaptchaSolver(
            yescaptcha_key,
            endpoint=endpoint,
            timeout=float(os.environ.get("GROK2API_YESCAPTCHA_TIMEOUT", "180") or 180),
            debug=True,
            on_progress=_turnstile_progress,
            auto_fallback_endpoint=True,
        )
        print(
            f"[grok-build-auth] turnstile website_url={website_url} "
            f"sitekey={sitekey} endpoint={getattr(solver, '_endpoint', '?')}"
        )
        try:
            turnstile = solver.solve_turnstile(
                website_url=website_url,
                website_key=sitekey,
                premium=True,
                fallback_non_premium=True,
            )
        except Exception as captcha_err:
            alt_url = "https://accounts.x.ai/sign-up?redirect=cloud-console"
            if website_url.rstrip("/") == alt_url.rstrip("/"):
                alt_url = "https://accounts.x.ai/sign-up?redirect=grok-com"
            update(
                "solving_turnstile",
                f"primary Turnstile failed ({captcha_err}); retry {alt_url}",
            )
            turnstile = solver.solve_turnstile(
                website_url=alt_url,
                website_key=sitekey,
                premium=False,
                fallback_non_premium=True,
            )
        if not turnstile:
            raise RuntimeError("YesCaptcha returned empty Turnstile token")
        update("creating_account", "Turnstile solved; creating xAI account")

        res = client.create_account(
            email=email,
            given_name="User",
            family_name="Grok",
            password=password,
            email_validation_code=code,
            turnstile_token=turnstile,
            castle_request_token="",
            conversion_id=str(uuid.uuid4()),
        )
        sc = list(getattr(res, "set_cookies", None) or [])
        rsc_body = getattr(res, "rsc_body", "") or ""
        rsc_preview = rsc_body[:800]
        print(f"[grok-build-auth] create_account HTTP={getattr(res, 'http_status', '?')}")
        print(f"[grok-build-auth] create_account set-cookies count={len(sc)}")
        print(f"[grok-build-auth] create_account rsc_body preview: {rsc_preview}")
        if not getattr(res, "ok", False):
            raise RuntimeError(
                "create_account failed / not confirmed. "
                f"HTTP {getattr(res, 'http_status', '?')}; "
                f"set_cookies={len(sc)}; body={rsc_preview!r}"
            )

        update("fetching_sso", "extracting SSO session")
        sso = client.fetch_sso_token(
            email=email, password=password, save=True, retries=6
        )
        if not sso:
            try:
                from xconsole_client.sso import (
                    SSOExtractor,
                    parse_sso_from_set_cookies,
                    parse_sso_token_from_text,
                )

                sso = parse_sso_from_set_cookies(sc) or parse_sso_token_from_text(
                    rsc_body
                )
                if not sso and rsc_body:
                    extractor = SSOExtractor(
                        transport_request=client._request,
                        base_headers=client._base_headers,
                        cookie_jar=client._t.cookies,
                        debug=True,
                    )
                    sso = extractor.extract(
                        rsc_body, email=email, password=password, save=False
                    )
            except Exception as recover_err:  # noqa: BLE001
                print(f"[grok-build-auth] SSO recover failed: {recover_err}")

        print(f"[grok-build-auth] fetch_sso_token result: {sso[:60] if sso else None}")
        sess["sso"] = sso
        session_cookies = extract_cookies_from_auth_client(client)
        print(
            f"[grok-build-auth] session cookies after signup: "
            f"{sorted((session_cookies or {}).keys())}"
        )
        if sso:
            session_cookies = dict(session_cookies or {})
            session_cookies["sso"] = sso
            session_cookies["sso-rw"] = sso

        if not sso:
            raise RuntimeError(
                "account step finished but SSO cookie was not obtained. "
                "Cannot continue to OAuth/CreateSession without browser session. "
                f"HTTP {getattr(res, 'http_status', '?')}, set_cookies={len(sc)}, "
                f"cookie_keys={sorted((session_cookies or {}).keys())}, "
                f"body_preview={rsc_preview!r}. "
                "Usually means create_account did not fully succeed, or the "
                "set-cookie hop is blocked by network/proxy/Cloudflare."
            )

        # Primary path: SSO cookie → OIDC device flow → auth.json
        update("importing", "SSO obtained; importing via device flow")
        sso_err: Exception | None = None
        try:
            import sso_to_auth_json as sso_import

            token = sso_import.sso_to_token(sso)
            if not token or not token.get("access_token"):
                raise RuntimeError(
                    "SSO device flow returned no access_token "
                    "(sso cookie may be invalid or device verify/approve failed)"
                )
            _key, entry = sso_import.token_to_auth_entry(token, email=email)
            import_result = accounts.import_auth_payload(
                {
                    "key": entry["key"],
                    "auth_mode": entry.get("auth_mode", "oidc"),
                    "email": entry.get("email") or email,
                    "refresh_token": entry.get("refresh_token", ""),
                    "expires_at": entry.get("expires_at"),
                    "oidc_issuer": entry.get("oidc_issuer", "https://auth.x.ai"),
                    "oidc_client_id": entry.get("oidc_client_id", ""),
                },
                merge=True,
            )
            if not import_result.get("ok"):
                raise RuntimeError(f"SSO import failed: {import_result.get('error')}")
            sess["auth_json"] = import_result
            sess["oauth"] = {
                "path": "sso_device_flow",
                "access_token": (token.get("access_token") or "")[:20] + "...",
                "refresh_token": bool(token.get("refresh_token")),
                "email": email,
            }
            update(
                "imported",
                f"imported via SSO device flow "
                f"({len(import_result.get('imported') or [])} account(s))",
            )
            return
        except Exception as e:  # noqa: BLE001
            sso_err = e
            update(
                "oauth",
                f"SSO device import failed ({e}); trying Build OAuth with SSO cookie",
            )

        update("oauth", "completing Build OAuth with SSO session cookie")
        try:
            oauth = xai_oauth_login_protocol(
                email,
                password,
                yescaptcha_key=yescaptcha_key,
                proxy=proxy or "",
                debug=True,
                turnstile_premium=True,
                cliproxyapi_auth_dir=None,
                cliproxyapi_base_url=UPSTREAM_BASE.rstrip("/"),
                cliproxyapi_disabled=True,
                output_dir=None,
                redirect_port=56121,
                session_cookies=session_cookies or None,
                auth_client=client,
            )
            sess["oauth"] = {
                "path": "build_oauth",
                "access_token": (
                    oauth.access_token[:20] + "..." if oauth.access_token else None
                ),
                "refresh_token": bool(oauth.refresh_token),
                "email": oauth.email,
            }

            update("importing", "importing Build OAuth token into auth.json")
            record = build_cliproxyapi_auth_record(
                oauth.token,
                userinfo=oauth.userinfo,
                redirect_uri=oauth.redirect_uri,
                disabled=False,
                base_url=UPSTREAM_BASE.rstrip("/"),
                headers=dict(CLIPROXYAPI_GROK_HEADERS),
            )
            import_result = accounts.import_auth_payload(
                {
                    "key": record["access_token"],
                    "auth_mode": "build_oauth",
                    "email": record.get("email") or email,
                    "refresh_token": record.get("refresh_token"),
                    "expires_at": record.get("expired"),
                    "oidc_issuer": "https://auth.x.ai",
                    "oidc_client_id": record.get("client_id", ""),
                    "first_name": record.get("first_name"),
                    "last_name": record.get("last_name"),
                    "principal_type": record.get("principal_type"),
                },
                merge=True,
            )
            sess["auth_json"] = import_result
            if not import_result.get("ok"):
                raise RuntimeError(f"import failed: {import_result.get('error')}")
            update(
                "imported",
                f"imported via Build OAuth "
                f"({len(import_result.get('imported') or [])} account(s))",
            )
            return
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "registration produced SSO but import failed. "
                f"SSO device flow: {sso_err}; Build OAuth: {e}. "
                f"sso_prefix={(sso or '')[:24]!r} "
                f"cookie_keys={sorted((session_cookies or {}).keys())}"
            ) from e
    except Exception as exc:  # noqa: BLE001
        update("error", f"failed: {exc}", error=str(exc))
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def list_registration_sessions() -> dict[str, Any]:
    _clean_old_sessions()
    return {"sessions": [_compact_session(s) for s in _sessions.values()]}


def get_registration_session(
    sid: str, *, include_auth_json: bool = False
) -> dict[str, Any] | None:
    sess = _sessions.get(sid)
    if not sess:
        return None
    out = dict(sess)
    out.pop("_client", None)
    out.pop("_oauth_client", None)
    if not include_auth_json:
        out.pop("auth_json", None)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    print("grok-build-auth adapter for grokcli-2api")
    result = start_registration()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("ok"):
        return 1

    sid = result["id"]
    deadline = time.time() + 600
    while time.time() < deadline:
        sess = get_registration_session(sid, include_auth_json=True)
        if not sess:
            print("session disappeared", file=sys.stderr)
            return 1
        status = sess.get("status")
        print(f"[{time.strftime('%H:%M:%S')}] {status}: {sess.get('message')}")
        if status in ("imported", "error"):
            print(json.dumps(sess, ensure_ascii=False, indent=2))
            return 0 if status == "imported" else 1
        time.sleep(5)

    print("timeout", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
