import tomllib
from pathlib import Path
from pydantic import BaseModel, Field

class PathsConfig(BaseModel):
    input_dir: str = "input"
    output_dir: str = "output"
    darktable_cli_path: str = "darktable-cli"

class ProcessingConfig(BaseModel):
    color_temperature: int = 3800
    exposure_boost_ev: float = 0.2
    clahe_clip_limit: float = 1.5
    clahe_tile_grid_size: int = 8
    sky_threshold: int = 30
    mask_blur_radius: int = 21
    saturation_boost: float = 1.3
    milky_way_brightness_multiplier: float = 0.8

class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

def load_config(config_path: str = "options.toml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        # Return default if not exists
        return AppConfig()
        
    with open(path, "rb") as f:
        data = tomllib.load(f)
        
    return AppConfig(**data)
