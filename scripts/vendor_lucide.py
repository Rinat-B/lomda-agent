"""Download a fixed subset of Lucide icons into agent/assets/lucide/ (run once; the result is committed).

    .venv\\Scripts\\python scripts/vendor_lucide.py

The motif is built from these icons only (agent/motif.py), so the lesson never loads anything from the
network and no model writes SVG paths. Lucide is ISC licensed; LICENSE is kept next to the icons.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from agent.svg_clean import is_clean

VERSION = "0.460.0"
BASE = f"https://unpkg.com/lucide-static@{VERSION}"
OUT = Path(__file__).resolve().parent.parent / "agent" / "assets" / "lucide"
# General subjects of training and compliance lessons: vehicles and weather, safety, people, rules.
ICONS = [
    "car", "truck", "bus", "bike", "route", "gauge", "timer", "clock", "calendar-check",
    "cloud-fog", "cloud-rain", "cloud-snow", "snowflake", "sun", "wind", "droplets", "thermometer",
    "triangle-alert", "shield-check", "shield", "siren", "flame", "hard-hat", "heart-pulse", "life-buoy",
    "eye", "hand", "lightbulb", "book-open", "graduation-cap", "clipboard-check", "file-check",
    "scale", "lock", "key-round", "users", "user-check", "handshake", "message-circle", "phone",
    "building-2", "warehouse", "package", "forklift", "wrench", "badge-check", "circle-check", "flag",
]


def inner(svg: str) -> str:
    """The drawing elements inside <svg>, without the comment and the outer element."""
    body = re.sub(r"<!--.*?-->", "", svg, flags=re.S)
    body = body[body.index(">", body.index("<svg")) + 1:body.rindex("</svg>")]
    return " ".join(body.split())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    icons, missing, rejected = {}, [], []
    with httpx.Client(timeout=30) as client:
        for name in ICONS:
            r = client.get(f"{BASE}/icons/{name}.svg")
            if r.status_code != 200:
                missing.append(name)
                continue
            body = inner(r.text)
            if not is_clean(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">{body}</svg>'):
                rejected.append(name)  # e.g. an embedded image instead of paths: not kept
                continue
            icons[name] = body
        (OUT / "LICENSE").write_text(client.get(f"{BASE}/LICENSE").text, encoding="utf-8")
    data = {"source": f"lucide-static@{VERSION}", "viewBox": "0 0 24 24", "icons": icons}
    (OUT / "icons.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(icons)} icons; missing: {missing}; rejected by the SVG cleaner: {rejected}")


if __name__ == "__main__":
    main()
