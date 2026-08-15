"""Package demo data into a downloadable zip for GitHub users."""

from __future__ import annotations

import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "examples" / "demo_data"
OUTPUT_ZIP = PROJECT_ROOT / "examples" / "demo_data.zip"


def package_demo_data() -> Path:
    if not DEMO_DIR.exists():
        raise FileNotFoundError(f"Demo data directory not found: {DEMO_DIR}")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(DEMO_DIR.rglob("*")):
            if file_path.is_file():
                arcname = Path("demo_data") / file_path.relative_to(DEMO_DIR)
                zf.write(file_path, arcname)

    print(f"Created {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size:,} bytes)")
    return OUTPUT_ZIP


if __name__ == "__main__":
    package_demo_data()
