r"""Copy a fixed subset of Lucide icons from the npm package into agent/assets/lucide/ (run once; the
result is committed, so the agent and the lessons never load icons from the network).

    .venv\Scripts\python scripts/vendor_lucide.py

The source is the npm package lucide-static (a .tgz from the npm registry, checked against the
registry's sha512 integrity), not a CDN: on day 10 the CDNs served some icons as embedded raster
images. Every icon still goes through the SVG cleaner. Lucide is ISC licensed; LICENSE is copied.
"""

import base64
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from agent.svg_clean import is_clean  # noqa: E402

VERSION = "0.460.0"
REGISTRY = f"https://registry.npmjs.org/lucide-static/{VERSION}"
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


def package() -> tarfile.TarFile:
    with httpx.Client(timeout=60) as client:
        meta = client.get(REGISTRY).json()["dist"]
        data = client.get(meta["tarball"]).content
    algo, expected = meta["integrity"].split("-", 1)
    if base64.b64encode(hashlib.new(algo, data).digest()).decode() != expected:
        raise SystemExit("the downloaded package does not match the registry's integrity hash")
    return tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    icons, missing, rejected = {}, [], []
    with package() as tar:
        names = set(tar.getnames())
        for name in ICONS:
            member = f"package/icons/{name}.svg"
            if member not in names:
                missing.append(name)
                continue
            body = inner(tar.extractfile(member).read().decode("utf-8"))
            if not is_clean(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">{body}</svg>'):
                rejected.append(name)
                continue
            icons[name] = body
        (OUT / "LICENSE").write_text(tar.extractfile("package/LICENSE").read().decode("utf-8"), encoding="utf-8")
    data = {"source": f"npm lucide-static@{VERSION}", "viewBox": "0 0 24 24", "icons": icons}
    (OUT / "icons.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(icons)} icons; missing: {missing}; rejected by the SVG cleaner: {rejected}")


if __name__ == "__main__":
    main()
