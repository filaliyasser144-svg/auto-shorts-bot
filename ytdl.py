import os
import subprocess

COOKIES_FILE = "cookies.txt"

ATTEMPTS = [
    [],
    ["--remote-components", "ejs:github"],
    ["--extractor-args", "youtube:player_client=tv,web_safari,mweb"],
]


def base_cmd():
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--js-runtimes", "node",
        "--retries", "5",
        "--socket-timeout", "30",
    ]
    if os.path.exists(COOKIES_FILE) and os.path.getsize(COOKIES_FILE) > 100:
        cmd += ["--cookies", COOKIES_FILE]
    return cmd


def run_ytdlp(args):
    for i, extra in enumerate(ATTEMPTS, start=1):
        cmd = base_cmd() + extra + args
        print(f"محاولة التنزيل رقم {i}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        tail = "\n".join((result.stderr or "").strip().splitlines()[-6:])
        print(f"فشلت المحاولة {i}:\n{tail}")
    return False
