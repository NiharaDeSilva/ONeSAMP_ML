# config.py
import os
from dataclasses import dataclass

# ===== Base Paths =====
BASE_PATH = "/blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML"
# BASE_PATH = "/Users/suhashidesilva/Documents/Projects/ONeSAMP_ML"

OUTPUT_PATH = os.path.join(BASE_PATH, "output_tuning2")
TEMP_DIR = os.path.join(BASE_PATH, "temp")

# ===== Executables =====
POPULATION_GENERATOR = os.path.join(BASE_PATH, "build", "OneSamp")

# Optional future paths
SCALAR_PATH = os.path.join(BASE_PATH, "scalars")


@dataclass
class configClass:
    fileName: str = None
    sampleSize: int = None
    numLoci: int = None
    BASE_PATH: str = BASE_PATH
    OUTPUT_PATH: str = OUTPUT_PATH
    scalar_path: str = SCALAR_PATH

config = configClass()
