import subprocess
import os
from pathlib import Path
import cv2
import numpy as np
import tifffile
from src.config import AppConfig

class DarktableProcessor:
    def __init__(self, config: AppConfig):
        self.config = config

    def convert_to_tiff(self, raw_path: Path, tiff_path: Path) -> bool:
        """Uses darktable-cli to convert RAW to 16-bit TIFF"""
        # Ensure no truncated/leftover TIFF file exists from a previous interrupted run
        if tiff_path.exists():
            try:
                tiff_path.unlink()
            except OSError:
                pass
                
        cli_path = self.config.paths.darktable_cli_path
        
        command = [
            cli_path,
            str(raw_path.resolve()).replace('\\', '/'),
            str(tiff_path.resolve()).replace('\\', '/'),
            "--core",
            "--conf", "plugins/imageio/format/tiff/bpp=16"
        ]

        print(f"Converting to TIFF: {raw_path.name}")
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error processing {raw_path.name}: {result.stderr}")
                return False
            return True
        except FileNotFoundError:
            print(f"Error: Could not find darktable-cli at '{cli_path}'. Is it installed and in PATH?")
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

    def enhance_milky_way(self, img_rgb: np.ndarray) -> np.ndarray:
        """Applies enhancements to the sky part (Milky Way)."""
        # Base original 8-bit image for blending (keeps dark sky dark)
        base_8bit = (img_rgb / 256).astype(np.uint8)

        # 1. Process the image for the Milky Way (exposure + CLAHE)
        exposure_multiplier = 2 ** self.config.processing.exposure_boost_ev
        img_exposed = np.clip(img_rgb * exposure_multiplier, 0, 65535).astype(np.uint16)
        img_exposed_8bit = (img_exposed / 256).astype(np.uint8)

        lab = cv2.cvtColor(img_exposed_8bit, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(
            clipLimit=self.config.processing.clahe_clip_limit, 
            tileGridSize=(self.config.processing.clahe_tile_grid_size, self.config.processing.clahe_tile_grid_size)
        )
        cl = clahe.apply(l)
        
        # 2. Create the Milky Way isolation mask based on original lightness
        orig_lab = cv2.cvtColor(base_8bit, cv2.COLOR_RGB2LAB)
        orig_l, orig_a, orig_b = cv2.split(orig_lab)
        
        # Heavy blur to get the "structure" of the sky and ignore sharp stars/noise
        blur_l = cv2.GaussianBlur(orig_l, (151, 151), 0).astype(np.float32)
        
        # Normalize to 0-1 range using percentiles to ignore bright outliers (like large stars)
        # By setting l_min to the 50th percentile (median), we guarantee that at least half of the sky
        # (the darker parts outside the Milky Way) gets a mask value of exactly 0 and remains completely untouched.
        l_min = np.percentile(blur_l, 50)
        l_max = np.percentile(blur_l, 99)
        mw_mask = np.clip((blur_l - l_min) / (l_max - l_min + 1e-5), 0, 1)
        
        # Cube it so the falloff from the bright Milky Way to the dark sky is steeper
        mw_mask = mw_mask ** 3.0

        # 3. Blend the lightness: Dark sky = original L, Milky Way = CLAHE L
        final_l = (cl * mw_mask + orig_l * (1.0 - mw_mask)).astype(np.float32)
        
        # Apply brightness reduction specifically to the Milky Way area based on config
        brightness_adjust = 1.0 + (self.config.processing.milky_way_brightness_multiplier - 1.0) * mw_mask
        final_l = np.clip(final_l * brightness_adjust, 0, 255).astype(np.uint8)
        
        limg = cv2.merge((final_l, orig_a, orig_b))
        enhanced_8bit = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # 4. Color & Saturation Boost on the isolated Milky Way
        hsv = cv2.cvtColor(enhanced_8bit, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat_boost = 1.0 + (self.config.processing.saturation_boost - 1.0) * mw_mask
        hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_boost, 0, 255)
        enhanced_8bit = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 5. Color temp adjustment
        # For simplicity we assume color_temperature is around 3800, standard is ~5000.
        b_scale = 5000 / self.config.processing.color_temperature
        r_scale = self.config.processing.color_temperature / 5000
        
        enhanced_float = enhanced_8bit.astype(np.float32)
        # Apply temperature shift more strongly to the milky way structures
        enhanced_float[:,:,2] = np.clip(enhanced_float[:,:,2] * (1.0 + (b_scale - 1.0) * mw_mask), 0, 255) # Blue
        enhanced_float[:,:,0] = np.clip(enhanced_float[:,:,0] * (1.0 + (r_scale - 1.0) * mw_mask), 0, 255) # Red
        
        return enhanced_float.astype(np.uint8)

    def process_image(self, raw_path: Path, output_path: Path) -> bool:
        print(f"Processing {raw_path.name}...")
        tiff_path = output_path.with_suffix('.tif')
        
        # 1. Convert to TIFF
        if not self.convert_to_tiff(raw_path, tiff_path):
            return False
            
        print("Applying Milky Way enhancements...")
        # 2. Read TIFF
        img_rgb_16 = tifffile.imread(str(tiff_path))
        
        # 3. Create Sky Mask
        # Convert to 8-bit grayscale for thresholding
        gray = cv2.cvtColor((img_rgb_16 / 256).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Blur to remove noise
        blur_rad = self.config.processing.mask_blur_radius
        if blur_rad % 2 == 0: blur_rad += 1
        blurred = cv2.GaussianBlur(gray, (blur_rad, blur_rad), 0)
        
        # Threshold: sky is usually brighter than foreground
        _, mask = cv2.threshold(blurred, self.config.processing.sky_threshold, 255, cv2.THRESH_BINARY)
        
        # Refine mask (morphological closing then opening)
        kernel = np.ones((15,15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Blur mask for smooth blending
        mask = cv2.GaussianBlur(mask, (blur_rad*3, blur_rad*3), 0)
        mask_float = mask.astype(np.float32) / 255.0
        mask_float = np.stack([mask_float]*3, axis=2) # 3 channels
        
        # 4. Enhance Sky
        enhanced_sky = self.enhance_milky_way(img_rgb_16)
        
        # 5. Foreground (just convert original to 8-bit)
        foreground = (img_rgb_16 / 256).astype(np.uint8)
        
        # 6. Blend
        final_img = (enhanced_sky * mask_float + foreground * (1.0 - mask_float)).astype(np.uint8)
        
        # 7. Save
        # cv2 uses BGR, but we did everything in RGB
        final_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), final_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        # 8. Save comparison image
        comparison_dir = output_path.parent / "comparison"
        comparison_dir.mkdir(exist_ok=True)
        
        orig_bgr = cv2.cvtColor(foreground, cv2.COLOR_RGB2BGR)
        comparison_img = cv2.hconcat([orig_bgr, final_bgr])
        
        comparison_path = comparison_dir / f"{raw_path.stem}_comparison.jpg"
        cv2.imwrite(str(comparison_path), comparison_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        
        # Cleanup temp TIFF
        try:
            tiff_path.unlink()
        except OSError:
            pass
            
        print(f"Successfully processed {raw_path.name} -> {output_path.name}")
        return True

    def run_batch(self):
        input_dir = Path(self.config.paths.input_dir)
        output_dir = Path(self.config.paths.output_dir)

        if not input_dir.exists():
            input_dir.mkdir(parents=True, exist_ok=True)
            
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        raw_files = list(set(list(input_dir.glob("*.arw")) + list(input_dir.glob("*.ARW"))))

        if not raw_files:
            print(f"No .arw files found in '{input_dir}'. Please add some test files.")
            return

        print(f"Found {len(raw_files)} raw files to process.")

        for raw_file in raw_files:
            output_file = output_dir / f"{raw_file.stem}_processed.jpg"
            self.process_image(raw_file, output_file)
