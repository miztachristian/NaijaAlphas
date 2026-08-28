"""
Standalone tests. Run before first deployment.

  python notify/test_notify.py            # sends test message
  python notify/test_notify.py --brief    # sends full brief
  python notify/test_notify.py --agent    # test AI agent (requires Ollama)
  python notify/test_notify.py --schema   # print verified column names
"""
import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()


def test_send():
    from notify.sender import send_telegram
    ok = send_telegram("✅ NGX bot connected successfully!")
    print("send_telegram:", "OK" if ok else "FAILED — check token/chat_id in .env")


def test_brief():
    from notify.formatter import build_brief
    from notify.sender import send_telegram
    admin_id = os.getenv("TELEGRAM_ADMIN_ID", os.getenv("TELEGRAM_CHAT_ID", ""))
    brief = build_brief(admin_id)
    print(f"Brief: {len(brief)} chars")
    print(brief[:800])
    print("---")
    ok = send_telegram(brief, chat_id=admin_id)
    print("send brief:", "OK" if ok else "FAILED")


def test_agent():
    from notify.agent import StockAgent
    admin_id = os.getenv("TELEGRAM_ADMIN_ID", os.getenv("TELEGRAM_CHAT_ID", ""))
    agent = StockAgent()
    print("Querying agent…")
    answer = agent.ask("What are my top 3 positions by current value?", user_id=admin_id)
    print("Answer:", answer)


def test_schema():
    from notify.tools import (
        get_decision_table, get_gamble_punts, get_momentum_picks, get_seasonality
    )
    admin_id = os.getenv("TELEGRAM_ADMIN_ID", os.getenv("TELEGRAM_CHAT_ID", "0"))
    dt = get_decision_table()
    print("decision_table cols:", list(dt.columns) if not dt.empty else "EMPTY")
    picks = get_momentum_picks(admin_id)
    print("momentum_picks[0]:", picks[0] if picks else "EMPTY")
    seas = get_seasonality()
    print("seasonality[0]:", seas[0] if seas else "EMPTY")
    punts = get_gamble_punts()
    print("punt cards:", len(punts), "| keys:", list(punts[0].keys()) if punts else "EMPTY")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--schema", action="store_true")
    args = ap.parse_args()
    if args.brief:
        test_brief()
    elif args.agent:
        test_agent()
    elif args.schema:
        test_schema()
    else:
        test_send()
