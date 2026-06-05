import sys
from pathlib import Path

# Add project root to path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config
from src.processor import DarktableProcessor


def main():
    print("Auto Milky Way - Starting Processor")
    config = load_config()
    
    print(f"Loaded config: Input Dir = {config.paths.input_dir}, Output Dir = {config.paths.output_dir}")
    
    processor = DarktableProcessor(config)
    processor.run_batch()

if __name__ == "__main__":
    main()
