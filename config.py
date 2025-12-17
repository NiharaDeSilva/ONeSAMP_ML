from dataclasses import dataclass

@dataclass
class Config:
    fileName: str = None
    sampleSize: int = None
    numLoci: int = None
    BASE_PATH: str = None
    output_path: str = None
    scalar_path: str = None

config = Config()
