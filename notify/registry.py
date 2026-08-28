"""
notify/registry.py — Multi-user store.
Storage: data/profiles/registry.json
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _path() -> Path:
    root = Path(__file__).resolve().parent.parent
    d = root / "data" / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    return d / "registry.json"


def _load() -> dict:
    p = _path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict) -> None:
    _path().write_text(json.dumps(data, indent=2), encoding="utf-8")


def _admin_id() -> str:
    return str(os.getenv("TELEGRAM_ADMIN_ID", os.getenv("TELEGRAM_CHAT_ID", "")))


def is_registered(user_id) -> bool:
    entry = _load().get(str(user_id))
    return entry is not None and entry.get("approved", False)


def is_admin(user_id) -> bool:
    return str(user_id) == _admin_id()


def register(user_id, name: str, username: str, auto_approve: bool = False) -> str:
    uid = str(user_id)
    reg = _load()
    if uid in reg:
        return "already_registered"
    approved = auto_approve or (uid == _admin_id())
    reg[uid] = {
        "name": name,
        "username": username,
        "registered_at": datetime.now().isoformat(),
        "is_admin": uid == _admin_id(),
        "approved": approved,
    }
    _save(reg)
    return "registered" if approved else "pending"


def approve(user_id) -> bool:
    uid = str(user_id)
    reg = _load()
    if uid not in reg:
        return False
    reg[uid]["approved"] = True
    _save(reg)
    return True


def list_users() -> list[dict]:
    return [{"user_id": uid, **v} for uid, v in _load().items()]


def get_user(user_id) -> Optional[dict]:
    return _load().get(str(user_id))
