from pathlib import Path
import os

PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / 'data'
print( Path(__file__).parent.resolve())

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)