#!/usr/bin/env python3
"""
批量导入 xAI SSO cookie 到项目 auth.json（纯 HTTP Device Flow）

用法:
  # 单个 / 批量 SSO，每个导入后按 user_id 合并到 data/auth.json
  python3 sso_to_auth_json.py --sso sso_list.txt

  # 写出多个独立 auth 文件（每个可直接 cp 到 ~/.grok/auth.json）
  python3 sso_to_auth_json.py --sso sso_list.txt --out-dir ./auth_out

  # 合并到指定 json（key 带 user_id 后缀，避免覆盖）
  python3 sso_to_auth_json.py --sso sso_list.txt --out auth_merged.json --merge

  # 单行 sso
  python3 sso_to_auth_json.py --sso-cookie 'eyJ...'

环境变量:
  GROK2API_AUTH_FILE  - 导入目标 auth.json（默认项目 data/auth.json）
  GROK2API_PROXY      - 代理地址，例如 http://127.0.0.1:7890
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from curl_cffi import requests

# Use project config when available, otherwise fall back to defaults
try:
    from config import AUTH_FILE, GROK_CLI_CLIENT_ID, OIDC_ISSUER, OIDC_SCOPES
except Exception:  # pragma: no cover - standalone fallback
    AUTH_FILE = Path(os.getenv("GROK2API_AUTH_FILE", str(Path.home() / ".grok" / "auth.json")))
    GROK_CLI_CLIENT_ID = os.getenv("GROK2API_OIDC_CLIENT_ID", "b1a00492-073a-47ea-816f-4c329264a828")
    OIDC_ISSUER = os.getenv("GROK2API_OIDC_ISSUER", "https://auth.x.ai")
    OIDC_SCOPES = os.getenv(
        "GROK2API_OIDC_SCOPES",
        "openid profile email offline_access grok-cli:access "
        "api:access conversations:read conversations:write",
    )

AUTH_KEY = f"{OIDC_ISSUER}::{GROK_CLI_CLIENT_ID}"


def _proxy_kwargs() -> dict:
    """Return curl_cffi compatible proxy kwargs from env."""
    proxy = os.getenv("GROK2API_PROXY") or os.getenv("GROK_CLI_PROXY") or ""
    if proxy:
        return {"proxies": {"http": proxy, "https": proxy}}
    return {}


def b64url_decode(seg: str) -> bytes:
    seg += "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg)


def decode_jwt_payload(token: str) -> dict:
    try:
        return json.loads(b64url_decode(token.split(".")[1]))
    except Exception:
        return {}


def rfc3339_ns(ts: float | None = None) -> str:
    """2026-07-10T01:00:00.000000000Z"""
    if ts is None:
        ts = time.time()
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + ".000000000Z"


def request_device_code() -> dict | None:
    data = urllib.parse.urlencode({"client_id": GROK_CLI_CLIENT_ID, "scope": OIDC_SCOPES}).encode()
    req = urllib.request.Request(
        f"{OIDC_ISSUER}/oauth2/device/code",
        data=data,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"  ❌ device/code HTTP {e.code}: {e.read().decode()[:200]}")
        return None


def poll_token(device_code: str, interval: int, expires_in: int, timeout: int = 60) -> dict | None:
    deadline = time.time() + min(expires_in, timeout)
    while time.time() < deadline:
        time.sleep(interval)
        data = urllib.parse.urlencode(
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": GROK_CLI_CLIENT_ID,
                "device_code": device_code,
            }
        ).encode()
        req = urllib.request.Request(
            f"{OIDC_ISSUER}/oauth2/token",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            err = json.loads(e.read())
            error = err.get("error", "")
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval += 5
                continue
            print(f"  ❌ token: {error}")
            return None
    print("  ❌ 轮询超时")
    return None


def sso_to_token(sso_cookie: str) -> dict | None:
    """SSO cookie → token dict (access/refresh/expires_in)"""
    s = requests.Session()
    s.cookies.set("sso", sso_cookie, domain=".x.ai")

    try:
        r = s.get("https://accounts.x.ai/", impersonate="chrome", timeout=15, **_proxy_kwargs())
    except Exception as e:
        print(f"  ❌ 网络错误: {e}")
        return None
    if "sign-in" in r.url or "sign-up" in r.url:
        print("  ❌ sso 无效")
        return None
    print("  ✅ sso 有效")

    print("  🔑 Device Flow...")
    dc = request_device_code()
    if not dc:
        return None
    print(f"  📋 user_code: {dc.get('user_code')}")

    try:
        s.get(dc["verification_uri_complete"], impersonate="chrome", timeout=15, **_proxy_kwargs())
        r = s.post(
            f"{OIDC_ISSUER}/oauth2/device/verify",
            data={"user_code": dc["user_code"]},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            impersonate="chrome",
            timeout=15,
            allow_redirects=True,
            **_proxy_kwargs(),
        )
        if "consent" not in r.url:
            print(f"  ❌ verify 失败: {r.url}")
            return None
    except Exception as e:
        print(f"  ❌ verify 异常: {e}")
        return None

    try:
        r = s.post(
            f"{OIDC_ISSUER}/oauth2/device/approve",
            data={
                "user_code": dc["user_code"],
                "action": "allow",
                "principal_type": "User",
                "principal_id": "",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            impersonate="chrome",
            timeout=15,
            allow_redirects=True,
            **_proxy_kwargs(),
        )
        if "done" not in r.url:
            print(f"  ❌ approve 失败: {r.url}")
            return None
        print("  ✅ 授权确认")
    except Exception as e:
        print(f"  ❌ approve 异常: {e}")
        return None

    token = poll_token(
        dc["device_code"],
        dc.get("interval", 5),
        dc.get("expires_in", 1800),
    )
    if not token:
        return None
    print(
        f"  ✅ access_token (expires_in={token.get('expires_in')}s)"
        + (" + refresh_token" if token.get("refresh_token") else "")
    )
    return token


def token_to_auth_entry(token: dict, email: str = "") -> tuple[str, dict]:
    """
    返回 (top_level_key, entry)
    top_level_key 固定为 issuer::client_id（与 ~/.grok/auth.json 一致）
    """
    access = token.get("access_token") or token.get("key") or ""
    refresh = token.get("refresh_token") or ""
    payload = decode_jwt_payload(access)

    user_id = payload.get("sub") or payload.get("principal_id") or ""
    principal_id = payload.get("principal_id") or user_id
    principal_type = payload.get("principal_type") or "User"

    expires_in = int(token.get("expires_in") or 21600)
    if "exp" in payload:
        expires_at = rfc3339_ns(float(payload["exp"]))
    else:
        expires_at = rfc3339_ns(time.time() + expires_in)

    iat = payload.get("iat")
    create_time = rfc3339_ns(float(iat) if iat else time.time())

    entry = {
        "key": access,
        "auth_mode": "oidc",
        "create_time": create_time,
        "user_id": user_id,
        "email": email or payload.get("email") or "",
        "principal_type": principal_type,
        "principal_id": principal_id,
        "refresh_token": refresh,
        "expires_at": expires_at,
        "oidc_issuer": OIDC_ISSUER,
        "oidc_client_id": GROK_CLI_CLIENT_ID,
    }
    return AUTH_KEY, entry


def write_auth_json(path: Path, auth_key: str, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {auth_key: entry}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def merge_auth_json(path: Path, auth_key: str, entry: dict, unique: bool = True) -> None:
    """
    合并写入。unique=True 时 key 变成 issuer::client_id::user_id，避免多账号互相覆盖。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    key = auth_key
    if unique and entry.get("user_id"):
        key = f"{auth_key}::{entry['user_id']}"
    existing[key] = entry
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def import_into_project_auth(entry: dict) -> str:
    """Use project's account manager to merge entry into AUTH_FILE."""
    import accounts as _accounts

    # Build a single-entry payload; _normalize_entry will derive user_id/email/expires_at.
    payload = {
        "key": entry["key"],
        "auth_mode": entry.get("auth_mode", "oidc"),
        "email": entry.get("email", ""),
        "refresh_token": entry.get("refresh_token", ""),
        "expires_at": entry.get("expires_at"),
        "oidc_issuer": entry.get("oidc_issuer", OIDC_ISSUER),
        "oidc_client_id": entry.get("oidc_client_id", GROK_CLI_CLIENT_ID),
    }
    result = _accounts.import_auth_payload(payload, merge=True)
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "import failed")
    imported = result.get("imported", [])
    return imported[0] if imported else ""


def load_sso_list(path: str | None, single: str | None) -> list[tuple[str, str]]:
    """Return list of (email_or_name, sso_cookie) tuples."""
    if single:
        return [("", single.strip())]
    if not path:
        return []
    out: list[tuple[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        email = ""
        # 兼容 邮箱----密码----sso 或 邮箱:密码:sso
        if "----" in line:
            parts = line.split("----")
            email = parts[0].strip()
            line = parts[-1].strip()
        elif ":" in line and not line.startswith("eyJ"):
            parts = line.rsplit(":", 1)
            email = parts[0].strip()
            line = parts[-1].strip()
        out.append((email, line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="SSO cookie → grok auth.json (纯 HTTP)")
    ap.add_argument("--sso", metavar="FILE", help="sso 列表文件（一行一个 JWT，或 邮箱----密码----sso）")
    ap.add_argument("--sso-cookie", metavar="JWT", help="单个 sso cookie")
    ap.add_argument("--out", default=None, help="输出 auth.json 路径（单账号或 --merge）")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="批量时每个账号写一个 {user_id}.json（可直接 cp 到 ~/.grok/auth.json）",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="合并到 --out，key 用 issuer::client_id::user_id",
    )
    ap.add_argument(
        "--into-project",
        action="store_true",
        default=True,
        help=f"默认导入到项目 auth.json: {AUTH_FILE}",
    )
    ap.add_argument(
        "--no-into-project",
        dest="into_project",
        action="store_false",
        help="不导入项目 auth.json，仅 --out / --out-dir 输出",
    )
    ap.add_argument("--delay", type=int, default=0, help="每个间隔秒数")
    ap.add_argument("--email", default="", help="写入 entry.email（可选）")
    args = ap.parse_args()

    cookies = load_sso_list(args.sso, args.sso_cookie)
    if not cookies:
        ap.error("需要 --sso 或 --sso-cookie")

    if len(cookies) > 1 and not args.out_dir and not args.merge and not args.into_project:
        args.out_dir = args.out_dir or "./auth_out"
        print(f"批量模式默认 --out-dir {args.out_dir}")

    if args.out is None and args.out_dir is None and len(cookies) == 1 and not args.into_project:
        args.out = str(Path.home() / ".grok" / "auth.json")

    target = "项目 auth.json" if args.into_project else (args.out or args.out_dir or "stdout")
    print(f"🚀 SSO → auth.json: {len(cookies)} 个, target={target}, delay={args.delay}s")
    ok = 0
    fail = 0

def process_one_sso(
    index: int,
    email_hint: str,
    sso: str,
    *,
    args_email: str,
    into_project: bool,
    out_dir: Path | None,
    out: Path | None,
    merge: bool,
    total: int,
) -> dict[str, Any]:
    """Process a single SSO cookie. Thread-safe for independent accounts."""
    result: dict[str, Any] = {"index": index, "email_hint": email_hint, "sso_hint": sso[:12] + "..." if len(sso) > 12 else "..."}
    try:
        token = sso_to_token(sso)
        if not token:
            result["status"] = "failed"
            result["error"] = "device flow failed or invalid sso"
            return result
        key, entry = token_to_auth_entry(token, email=args_email or email_hint)
        uid = entry.get("user_id") or secrets.token_hex(4)

        if out_dir:
            p = out_dir / f"{uid}.json"
            write_auth_json(p, key, entry)
            result["wrote"] = str(p)
        if out:
            if merge or total > 1:
                merge_auth_json(out, key, entry, unique=True)
                result["merged"] = str(out)
            else:
                write_auth_json(out, key, entry)
                result["wrote"] = str(out)
        if into_project:
            aid = import_into_project_auth(entry)
            result["imported_key"] = aid

        result["status"] = "ok"
        result["user_id"] = uid
        result["email"] = entry.get("email")
        return result
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        return result


def run_concurrent(
    cookies: list[tuple[str, str]],
    *,
    max_workers: int,
    delay: int,
    args_email: str,
    into_project: bool,
    out_dir: Path | None,
    out: Path | None,
    merge: bool,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Run SSO imports concurrently with per-item delay handled inside threads."""
    results: list[dict[str, Any]] = [None] * len(cookies)
    ok = 0
    fail = 0

    def _worker(args: tuple[int, str, str]) -> tuple[int, dict[str, Any]]:
        i, email_hint, sso = args
        if delay > 0 and i > 1:
            time.sleep(delay * (i - 1))
        res = process_one_sso(
            i,
            email_hint,
            sso,
            args_email=args_email,
            into_project=into_project,
            out_dir=out_dir,
            out=out,
            merge=merge,
            total=len(cookies),
        )
        print(
            f"\n{'=' * 60}\n[{i}/{len(cookies)}] {email_hint or ''}\n{'=' * 60}"
        )
        for k, v in res.items():
            if k in ("index", "email_hint", "sso_hint"):
                continue
            if k == "status":
                mark = "✅" if v == "ok" else "❌"
                print(f"  {mark} [{i}] {v}")
            elif isinstance(v, str):
                print(f"  💾 {k}: {v}")
            else:
                print(f"  • {k}: {v}")
        return i - 1, res

    workers = min(max_workers, max(1, len(cookies)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sso-") as ex:
        for idx, res in ex.map(_worker, ((i, e, s) for i, (e, s) in enumerate(cookies, 1))):
            results[idx] = res
            if res.get("status") == "ok":
                ok += 1
            else:
                fail += 1

    return ok, fail, results


def main() -> int:
    ap = argparse.ArgumentParser(description="SSO cookie → grok auth.json (纯 HTTP)")
    ap.add_argument("--sso", metavar="FILE", help="sso 列表文件（一行一个 JWT，或 邮箱----密码----sso）")
    ap.add_argument("--sso-cookie", metavar="JWT", help="单个 sso cookie")
    ap.add_argument("--out", default=None, help="输出 auth.json 路径（单账号或 --merge）")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="批量时每个账号写一个 {user_id}.json（可直接 cp 到 ~/.grok/auth.json）",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="合并到 --out，key 用 issuer::client_id::user_id",
    )
    ap.add_argument(
        "--into-project",
        action="store_true",
        default=True,
        help=f"默认导入到项目 auth.json: {AUTH_FILE}",
    )
    ap.add_argument(
        "--no-into-project",
        dest="into_project",
        action="store_false",
        help="不导入项目 auth.json，仅 --out / --out-dir 输出",
    )
    ap.add_argument("--delay", type=int, default=0, help="每个间隔秒数")
    ap.add_argument("--email", default="", help="写入 entry.email（可选）")
    ap.add_argument(
        "--threads",
        type=int,
        default=4,
        help="并发线程数（默认 4）",
    )
    args = ap.parse_args()

    cookies = load_sso_list(args.sso, args.sso_cookie)
    if not cookies:
        ap.error("需要 --sso 或 --sso-cookie")

    if len(cookies) > 1 and not args.out_dir and not args.merge and not args.into_project:
        args.out_dir = args.out_dir or "./auth_out"
        print(f"批量模式默认 --out-dir {args.out_dir}")

    if args.out is None and args.out_dir is None and len(cookies) == 1 and not args.into_project:
        args.out = str(Path.home() / ".grok" / "auth.json")

    target = "项目 auth.json" if args.into_project else (args.out or args.out_dir or "stdout")
    print(f"🚀 SSO → auth.json: {len(cookies)} 个, target={target}, delay={args.delay}s, threads={args.threads}")

    ok, fail, results = run_concurrent(
        cookies,
        max_workers=args.threads,
        delay=args.delay,
        args_email=args.email,
        into_project=args.into_project,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        out=Path(args.out) if args.out else None,
        merge=args.merge,
    )

    print(f"\n{'=' * 60}\n📊 完成: {ok}/{len(cookies)} 成功, {fail} 失败")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
