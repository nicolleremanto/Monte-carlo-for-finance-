"""Liste les modules Python chargés par l'interface dans Pyodide (app/manifest.json)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build() -> dict:
    files = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "mcfin").rglob("*.py"))
    return {"packages": ["numpy", "scipy"], "files": files}


if __name__ == "__main__":
    (ROOT / "app" / "manifest.json").write_text(json.dumps(build(), indent=1) + "\n")
    print("app/manifest.json mis à jour")
