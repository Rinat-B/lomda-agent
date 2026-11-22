"""Output guard for SVG (the motif, and any SVG that goes into lesson.json): an allow-list cleaner.

Only drawing elements and presentation attributes are kept. Removed: scripts, event attributes
(on*), foreignObject, embedded images and frames, links (href / xlink:href) except to an id in the
same SVG, style or attribute values with url() or javascript:, and anything outside the allow
lists. A document with a DOCTYPE or entities is refused (no entity expansion). The result is
re-serialised from the parsed tree, so what is kept is exactly what was checked.
"""

import re
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ALLOWED_ELEMENTS = {"svg", "g", "path", "circle", "ellipse", "rect", "line", "polyline", "polygon",
                    "title", "desc", "defs", "lineargradient", "radialgradient", "stop", "clippath", "use"}
ALLOWED_ATTRIBUTES = {
    "xmlns", "viewbox", "width", "height", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry",
    "d", "points", "fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "stroke-dasharray",
    "opacity", "fill-opacity", "stroke-opacity", "transform", "id", "class", "role", "aria-hidden",
    "aria-label", "focusable", "offset", "stop-color", "stop-opacity", "gradientunits", "clip-path", "href",
    "preserveaspectratio",
}
_UNSAFE_VALUE = re.compile(r"url\s*\(\s*['\"]?(?!#)|javascript:|data:|expression\s*\(", re.I)
_DOCTYPE = re.compile(r"<!DOCTYPE|<!ENTITY", re.I)


class UnsafeSvg(ValueError):
    """The SVG cannot be parsed safely."""


def _local(name: str) -> str:
    return name.rsplit("}", 1)[-1].lower()


def clean_svg(svg: str) -> tuple[str, list[str]]:
    """(clean SVG, what was removed). Raises UnsafeSvg for a DOCTYPE/entities or invalid XML."""
    if _DOCTYPE.search(svg or ""):
        raise UnsafeSvg("a DOCTYPE or entity declaration")
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        raise UnsafeSvg(f"not valid XML ({e})") from e
    if _local(root.tag) != "svg":
        raise UnsafeSvg("the root element is not <svg>")
    removed: list[str] = []

    def walk(element) -> None:
        for child in list(element):
            name = _local(child.tag) if isinstance(child.tag, str) else ""
            if name not in ALLOWED_ELEMENTS:
                element.remove(child)
                removed.append(f"<{name or 'comment'}>")
                continue
            walk(child)
        for attr in list(element.attrib):
            name, value = _local(attr), element.attrib[attr]
            if name.startswith("on"):
                removed.append(f"{name}=")
                del element.attrib[attr]
            elif name == "href" and not value.startswith("#"):
                removed.append(f"href={value[:30]}")
                del element.attrib[attr]
            elif name == "style" or name not in ALLOWED_ATTRIBUTES:
                removed.append(f"{name}=")
                del element.attrib[attr]
            elif _UNSAFE_VALUE.search(value):
                removed.append(f"{name}={value[:30]}")
                del element.attrib[attr]

    walk(root)
    for attr in list(root.attrib):  # the root too
        name = _local(attr)
        if name.startswith("on") or name == "style" or name not in ALLOWED_ATTRIBUTES:
            removed.append(f"{name}=")
            del root.attrib[attr]
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    return ET.tostring(root, encoding="unicode"), removed


def is_clean(svg: str) -> bool:
    try:
        return not clean_svg(svg)[1]
    except UnsafeSvg:
        return False
