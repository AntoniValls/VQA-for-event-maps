import sys


def progress_bar(pct, width=30):
    pct = max(0.0, min(100.0, float(pct)))
    filled = int(width * pct / 100.0)
    bar = "█" * filled + "·" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {pct:6.2f}%")
    sys.stdout.flush()
