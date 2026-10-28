import os

# ---- secrets (set them as GitHub repository secrets) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---- models (names change over time: check each provider's docs, override via env) ----
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ---- clip rules ----
CLIP_MIN = 25          # seconds
CLIP_MAX = 60
WINDOW_SEC = 1200      # transcript is sent to Gemini in 20-minute windows
WINDOW_OVERLAP = 60
CHUNK_SEC = 600        # audio is sent to Whisper in 10-minute chunks

WORK_DIR = "work"
OUT_DIR = "out"
