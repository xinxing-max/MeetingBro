import importlib
import os

mods = [
    "faster_whisper",
    "sounddevice",
    "soundfile",
    "numpy",
    "aiosqlite",
    "openai",
    "anthropic",
]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"{m} OK")
    except Exception:
        print(f"{m} MISSING")

for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    print(f"{key} {'SET' if os.environ.get(key) else 'UNSET'}")
