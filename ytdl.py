import os
import subprocess

COOKIES_FILE = "cookies.txt"

ATTEMPTS = [
    (True,  []),
    (True,  ["--remote-components", "ejs:github"]),
    (True,  ["--extractor-args", "youtube:player_client=tv,web_safari"]),
    (False, ["--extractor-args", "youtube:player_client=android_vr"]),
    (False, []),
]


def base_cmd(use_cookies):
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--js-runtimes", "node",
        "--retries", "3",
        "--socket-timeout", "30",
    ]
    if use_cookies and os.path.exists(COOKIES_FILE) and os.path.getsize(COOKIES_FILE) > 100:
        cmd += ["--cookies", COOKIES_FILE]
    return cmd


def run_ytdlp(args):
    for i, (use_cookies, extra) in enumerate(ATTEMPTS, start=1):
        cmd = base_cmd(use_cookies) + extra + args
        print(f"محاولة التنزيل رقم {i} (كوكيز: {use_cookies})...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        tail = "\n".join((result.stderr or "").strip().splitlines()[-8:])
        print(f"فشلت المحاولة {i}:\n{tail}")
    return False
