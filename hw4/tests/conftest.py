import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_tmp = Path(tempfile.mkdtemp())
shutil.copytree(ROOT / "prompts", _tmp / "prompts")
os.environ["MOCK_LLM"] = "1"
os.environ["PROMPT_DIR"] = str(_tmp / "prompts")
os.environ["DATA_DIR"] = str(_tmp / "data")
