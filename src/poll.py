import json
import os
import re
import urllib.parse
import urllib.request

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
MANUAL = (os.environ.get("INPUT_URL") or "").strip()
API = f"https://api.telegram.org/bot{TOKEN}"

YT = re.compile(
    r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?[^\s]*v=|live/|shorts/)|youtu\.be/)[\w\-]+[^\s]*"
)


def call(method, **params):
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(f"{API}/{method}", data=data, timeout=30) as r:
        return json.load(r)


def set_output(url):
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        f.write(f"url={url}\n")


def main():
    if not ALLOWED:
        print("TELEGRAM_CHAT_ID غير مضبوط.")
        set_output("")
        return

    if MANUAL:
        call("sendMessage", chat_id=ALLOWED, text="⏳ استلمت الرابط، بدأت المعالجة...")
        set_output(MANUAL)
        return

    updates = call("getUpdates", timeout=0).get("result", [])
    if not updates:
        set_output("")
        return

    last_id = updates[-1]["update_id"]
    call("getUpdates", offset=last_id + 1, limit=1)

    found = ""
    for u in updates:
        msg = u.get("message") or {}
        if str(msg.get("chat", {}).get("id")) != ALLOWED:
            continue
        m = YT.search(msg.get("text", ""))
        if m:
            found = m.group(0)

    if found:
        call("sendMessage", chat_id=ALLOWED, text="⏳ استلمت الرابط، بدأت المعالجة...")
    set_output(found)


if __name__ == "__main__":
    main()
