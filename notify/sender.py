"""
notify/sender.py
================
Telegram Bot API wrapper. Escapes HTML, truncates safely at newline boundaries.
Never raises — a delivery failure must not abort the decision run.
"""
from __future__ import annotations

import html
import logging
import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv(override=True)  # .env is the source of truth (see bot_server.py)
logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"
_LIMIT = 4096
_TIMEOUT = 15


def escape(text: str) -> str:
    """Escape &, <, > for Telegram HTML mode."""
    return html.escape(text, quote=False)


def _safe_truncate(text: str, limit: int = _LIMIT - 20) -> str:
    """
    Truncate at a newline boundary to avoid splitting HTML tags.
    Appends an ellipsis indicator.
    """
    if len(text) <= limit:
        return text
    cut = text[:limit].rfind("\n")
    if cut < limit // 2:
        cut = limit  # fallback: no newline found, hard cut
    return text[:cut] + "\n…"


def send_telegram(text: str,
                  chat_id: Optional[str] = None,
                  parse_mode: str = "HTML") -> bool:
    """Send text to Telegram. Returns True on success."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        return False

    text = _safe_truncate(text)
    url = _API.format(token=token)

    try:
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return True
        logger.error("Telegram %s: %s", resp.status_code, resp.text[:300])
        return False
    except Exception as exc:
        logger.error("send_telegram failed: %s", exc)
        return False
