import json
import os
import re
import urllib.parse
import urllib.request

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
MANUAL = (os.environ.get("INPUT_URL") or "").strip()
INPUT_MODE = (os.environ.get("INPUT_MODE") or "shorts").strip()
INPUT_MIN = (os.environ.get("INPUT_MIN") or "").strip()
API = f"https://api.telegram.org/bot{TOKEN}"

YT = re.compile(
    r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?[^\s]*v=|live/|shorts/)|youtu\.be/)[\w\-]+[^\s]*"
)
FOOTBALL = re.compile(r"(كورة|كرة|مباراة|ملخص|أهداف|اهداف|football|soccer|match|goals)", re.I)


def call(method, **params):
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(f"{API}/{method}", data=data, timeout=30) as r:
        return json.load(r)


def set_output(url, mode="shorts", minutes=""):
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        f.write(f"url={url}\nmode={mode}\nminutes={minutes}\n")


def parse(text):
    """رسالة عادية = شورتات. وإذا كتبت معها 'كورة' أو 'football' = ملخص مباراة (ويمكنك كتابة عدد الدقائق)."""
    m = YT.search(text)
    if not m:
        return None
    url = m.group(0)
    rest = text.replace(url, " ")
    if FOOTBALL.search(rest):
        n = re.search(r"\b(\d{1,2})\b", rest)
        return url, "football", (n.group(1) if n else "")
    return url, "shorts", ""


def main():
    if not ALLOWED:
        print("TELEGRAM_CHAT_ID غير مضبوط.")
        set_output("")
        return

    if MANUAL:
        mode = "football" if INPUT_MODE == "football" else "shorts"
        call("sendMessage", chat_id=ALLOWED, text="⏳ استلمت الرابط، بدأت المعالجة...")
        set_output(MANUAL, mode, INPUT_MIN if mode == "football" else "")
        return

    updates = call("getUpdates", timeout=0).get("result", [])
    if not updates:
        set_output("")
        return

    last_id = updates[-1]["update_id"]
    call("getUpdates", offset=last_id + 1, limit=1)

    found = None
    for u in updates:
        msg = u.get("message") or {}
        if str(msg.get("chat", {}).get("id")) != ALLOWED:
            continue
        p = parse(msg.get("text", ""))
        if p:
            found = p

    if found:
        url, mode, minutes = found
        note = "⏳ استلمت رابط المباراة، سألخّص أبرز اللقطات والأهداف..." if mode == "football" \
            else "⏳ استلمت الرابط، بدأت المعالجة..."
        call("sendMessage", chat_id=ALLOWED, text=note)
        set_output(url, mode, minutes)
    else:
        set_output("")


if __name__ == "__main__":
    main()
