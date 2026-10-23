import os
import sys
from pathlib import Path
from streamlit.web import cli as stcli

if getattr(sys, "frozen", False):
    app_root = Path(sys._MEIPASS)
else:
    app_root = Path(__file__).resolve().parent

os.chdir(app_root)

app_file = app_root / "frontend.py"

if not app_file.exists():
    raise FileNotFoundError(
        f"Could not find the Streamlit app: {app_file}"
    )

sys.argv = [
    "streamlit",
    "run",
    str(app_file),
    "--global.developmentMode=false",
]

raise SystemExit(stcli.main())