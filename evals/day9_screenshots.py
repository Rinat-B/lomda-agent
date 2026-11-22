"""Screenshots of a run's illustrations at 390 px, before and after using the first controls.

    .venv\\Scripts\\python -m evals.day9_screenshots evals/results/day9/free

Writes <run>/screenshots/<file>-before.png and -after.png (installed Edge, network blocked).
"""

import sys
from pathlib import Path


def shoot(run_dir: Path) -> list[Path]:
    from playwright.sync_api import sync_playwright

    from agent.illustrations import _use_first_control, wait_until_still

    pages = sorted((run_dir / "lesson" / "assets" / "illustrations").glob("*.html"))
    out = run_dir / "screenshots"
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        for page_path in pages:
            context = browser.new_context(offline=True, viewport={"width": 390, "height": 560})
            context.route("**/*", lambda r: r.continue_() if r.request.url.startswith("file:") else r.abort())
            page = context.new_page()
            page.goto(page_path.resolve().as_uri())
            wait_until_still(page)
            before = out / f"{page_path.stem}-before.png"
            page.screenshot(path=str(before), full_page=True)
            _use_first_control(page)
            wait_until_still(page)  # the "after" picture only when the animations ended
            after = out / f"{page_path.stem}-after.png"
            page.screenshot(path=str(after), full_page=True)
            saved += [before, after]
            context.close()
        browser.close()
    return saved


if __name__ == "__main__":
    for path in shoot(Path(sys.argv[1])):
        print(path)
