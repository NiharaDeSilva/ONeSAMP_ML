# config.py
import os
from dataclasses import dataclass

# ===== Base Paths =====
BASE_PATH = "/blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML"
# BASE_PATH = "/Users/suhashidesilva/Documents/Projects/ONeSAMP_ML"
OUTPUT = os.path.join(BASE_PATH, "Final")

OUTPUT_PATH = os.path.join(OUTPUT, "output_ml_100" )
TEMP_DIR = os.path.join(BASE_PATH, "temp")
PLOT_DIR = os.path.join(OUTPUT, "plots_ml_100")

# ===== Executables =====
POPULATION_GENERATOR = os.path.join(BASE_PATH, "build", "OneSamp")

# Optional future paths
SCALAR_PATH = os.path.join(BASE_PATH, "scalars")

@dataclass
class configClass:
    fileName: str = None
    sampleSize: int = None
    numLoci: int = None


