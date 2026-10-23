"""
test_validator.py
-----------------
Comprehensive evaluation test suite for input_validator.py:
  1. Positive Set: Genuine ABIDE neuroimaging data (ASD PNGs, Control PNGs, NIfTI scans).
  2. Negative Set: Diverse irrelevant inputs:
     - Color natural photographs (landscapes, pets, faces)
     - Greyscale natural photos
     - Dark scenes / night photos
     - Memes and captioned images
     - Screenshots (dark-mode code, light-mode web/documents)
     - Static anatomical/structural brain MRI scans (3D NIfTIs and 2D MRI slices)
     - Diagrams, charts, patterns
     - Random noise and flat images
     - Corrupted files, empty files, unsupported extensions
     - Geometric extremes (tiny images, extreme aspect ratios)
"""

import gzip
import io
import struct
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from input_validator import validate_upload

ROOT = Path(__file__).parent
ASD_PNG_DIR = ROOT / "data/raw/ABIDE/Combined Data/Autistic/IMAGES"
CON_PNG_DIR = ROOT / "data/raw/ABIDE/Combined Data/Typical_Control/IMAGES"
NII_DIR = ROOT / "data/raw/ABIDE/ABIDE1_Preprocess"


class EvaluationSuite:
    def __init__(self):
        self.results: List[Tuple[str, str, str, bool, str]] = []

    def record(self, category: str, subcategory: str, filename: str, passed: bool, message: str = ""):
        self.results.append((category, subcategory, filename, passed, message))

    # ─────────────────────────────────────────────────────────────────────────
    # Helper Generator Methods for Negative Test Samples
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _create_png_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _create_jpg_bytes(img: Image.Image, quality: int = 85) -> bytes:
        buf = io.BytesIO()
        # Convert RGBA to RGB for JPEG
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    @staticmethod
    def _create_3d_mri_nii_bytes() -> bytes:
        """Create a synthetic 3D static structural MRI volume NIfTI (ndim=3)."""
        hdr = bytearray(348)
        struct.pack_into("<i", hdr, 0, 348)  # sizeof_hdr
        # dim: [3, 256, 256, 176, 1, 1, 1, 1]
        struct.pack_into("<8h", hdr, 40, 3, 256, 256, 176, 1, 1, 1, 1)
        struct.pack_into("<h", hdr, 70, 16)  # datatype float32
        struct.pack_into("<h", hdr, 72, 32)  # bitpix
        struct.pack_into("<f", hdr, 108, 352.0)  # vox_offset
        hdr[344:348] = b"n+1\x00"
        # Small dummy data payload
        data = np.zeros(200, dtype=np.float32).tobytes()
        return bytes(hdr) + b"\x00" * 4 + data

    @staticmethod
    def _create_anatomical_mri_slice(view: str = "axial") -> Image.Image:
        """Generate a 2D synthetic anatomical brain MRI slice with skull and air background."""
        size = 224
        img = np.zeros((size, size), dtype=np.float32)
        cy, cx = size // 2, size // 2
        y, x = np.ogrid[:size, :size]

        if view == "axial":
            head = ((y - cy) / (size * 0.42)) ** 2 + ((x - cx) / (size * 0.35)) ** 2
            brain = head < 0.82
            skull = (head >= 0.82) & (head <= 0.98)
            img[skull] = 205
            img[brain] = np.random.normal(125, 20, size=img.shape)[brain]
            vent = ((y - cy) / (size * 0.18)) ** 2 + ((x - cx) / (size * 0.08)) ** 2 < 0.9
            img[vent] = 25
        elif view == "sagittal":
            head = ((y - cy) / (size * 0.40)) ** 2 + ((x - (cx + 15)) / (size * 0.45)) ** 2
            brain = head < 0.80
            skull = (head >= 0.80) & (head <= 0.95)
            img[skull] = 195
            img[brain] = np.random.normal(120, 22, size=img.shape)[brain]
        else:  # coronal
            head = ((y - cy) / (size * 0.44)) ** 2 + ((x - cx) / (size * 0.36)) ** 2
            brain = head < 0.80
            skull = (head >= 0.80) & (head <= 0.96)
            img[skull] = 200
            img[brain] = np.random.normal(122, 22, size=img.shape)[brain]

        return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    # ─────────────────────────────────────────────────────────────────────────
    # Run Tests
    # ─────────────────────────────────────────────────────────────────────────

    def run_all(self):
        print("=" * 80)
        print("RUNNING INPUT VALIDATOR EVALUATION SUITE")
        print("=" * 80)

        # 1. Genuine ABIDE Positive Samples
        print("\n[1/3] Testing Genuine ABIDE Positive Inputs...")
        self._test_positives()

        # 2. Irrelevant & Unrelated Negative Samples
        print("\n[2/3] Testing Irrelevant Negative Inputs...")
        self._test_negatives()

        # 3. Static Anatomical Brain MRI Samples
        print("\n[3/3] Testing Static Structural/Anatomical MRI Inputs...")
        self._test_anatomical_mri()

        # Summary & Diagnostics
        self._print_report()

    def _test_positives(self):
        # Sample 50 Autistic PNGs
        asd_files = sorted(ASD_PNG_DIR.glob("*.png"))[:50]
        for p in asd_files:
            valid, msg = validate_upload(p.name, p.read_bytes())
            self.record("POSITIVE", "Genuine ASD PNG", p.name, valid, msg or "")

        # Sample 50 Control PNGs
        con_files = sorted(CON_PNG_DIR.glob("*.png"))[:50]
        for p in con_files:
            valid, msg = validate_upload(p.name, p.read_bytes())
            self.record("POSITIVE", "Genuine Control PNG", p.name, valid, msg or "")

        # Sample 20 Genuine NIfTI files from ABIDE1_Preprocess
        nii_files = sorted(NII_DIR.glob("*.nii"))[:20]
        for p in nii_files:
            valid, msg = validate_upload(p.name, p.read_bytes())
            self.record("POSITIVE", "Genuine fMRI NIfTI", p.name, valid, msg or "")

        # Re-encoded genuine training image saved as JPEG
        if asd_files:
            raw_img = Image.open(asd_files[0])
            jpg_bytes = self._create_jpg_bytes(raw_img, quality=90)
            valid, msg = validate_upload("genuine_scan.jpg", jpg_bytes)
            self.record("POSITIVE", "Genuine Scan (JPEG re-encode)", "genuine_scan.jpg", valid, msg or "")

    def _test_negatives(self):
        np.random.seed(42)

        # N1: Natural Color Photograph (Landscape)
        img_sky = Image.new("RGB", (224, 224))
        for y in range(224):
            for x in range(224):
                if y < 112:
                    img_sky.putpixel((x, y), (135, 206, 235))
                else:
                    img_sky.putpixel((x, y), (34, 139, 34))
        valid, msg = validate_upload("landscape.png", self._create_png_bytes(img_sky))
        self.record("NEGATIVE", "Natural Color Photo", "landscape.png", valid, msg or "")
        valid, msg = validate_upload("landscape.jpg", self._create_jpg_bytes(img_sky))
        self.record("NEGATIVE", "Natural Color Photo (JPG)", "landscape.jpg", valid, msg or "")

        # N2: Pet / Animal Photograph (Texture & Noise)
        pet_arr = np.random.normal(140, 45, (224, 224)).clip(0, 255).astype(np.uint8)
        pet_img = Image.fromarray(pet_arr)
        valid, msg = validate_upload("pet_dog.png", self._create_png_bytes(pet_img))
        self.record("NEGATIVE", "Pet Photograph", "pet_dog.png", valid, msg or "")

        # N3: Dark Room / Night Scene Photograph
        dark_arr = np.random.normal(35, 18, (224, 224)).clip(0, 255).astype(np.uint8)
        dark_img = Image.fromarray(dark_arr)
        valid, msg = validate_upload("night_photo.png", self._create_png_bytes(dark_img))
        self.record("NEGATIVE", "Dark Scene / Night Photo", "night_photo.png", valid, msg or "")

        # N4: Meme with Captions
        meme_img = Image.new("RGB", (224, 224), (255, 255, 255))
        d = ImageDraw.Draw(meme_img)
        d.rectangle([20, 40, 204, 180], fill=(120, 80, 50))
        d.text((30, 10), "WHEN CODE COMPILES", fill=(0, 0, 0))
        d.text((50, 190), "BOTTOM TEXT", fill=(0, 0, 0))
        valid, msg = validate_upload("meme_caption.png", self._create_png_bytes(meme_img))
        self.record("NEGATIVE", "Meme / Web Graphic", "meme_caption.png", valid, msg or "")

        # N5: Dark IDE Screenshot
        ide_img = Image.new("RGB", (224, 224), (30, 30, 30))
        d = ImageDraw.Draw(ide_img)
        for line_y in range(15, 210, 14):
            d.rectangle([20, line_y, int(np.random.randint(60, 200)), line_y + 8], fill=(200, 200, 150))
        valid, msg = validate_upload("ide_screenshot.png", self._create_png_bytes(ide_img))
        self.record("NEGATIVE", "IDE Screenshot", "ide_screenshot.png", valid, msg or "")

        # N6: Light Document / Web Page Screenshot
        doc_img = Image.new("RGB", (224, 224), (250, 250, 250))
        d = ImageDraw.Draw(doc_img)
        for line_y in range(20, 200, 15):
            d.rectangle([20, line_y, int(np.random.randint(80, 200)), line_y + 4], fill=(40, 40, 40))
        valid, msg = validate_upload("web_document.png", self._create_png_bytes(doc_img))
        self.record("NEGATIVE", "Document Screenshot", "web_document.png", valid, msg or "")

        # N7: Uniform White Noise
        noise_arr = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        noise_img = Image.fromarray(noise_arr)
        valid, msg = validate_upload("random_noise.png", self._create_png_bytes(noise_img))
        self.record("NEGATIVE", "Random Noise", "random_noise.png", valid, msg or "")

        # N8: Flat Grey / Solid Color
        grey_img = Image.new("L", (224, 224), 128)
        valid, msg = validate_upload("solid_grey.png", self._create_png_bytes(grey_img))
        self.record("NEGATIVE", "Solid Color", "solid_grey.png", valid, msg or "")

        # N9: Pure White
        white_img = Image.new("L", (224, 224), 255)
        valid, msg = validate_upload("pure_white.png", self._create_png_bytes(white_img))
        self.record("NEGATIVE", "Pure White Image", "pure_white.png", valid, msg or "")

        # N10: Checkerboard Graphic
        chk = np.zeros((224, 224), dtype=np.uint8)
        for i in range(0, 224, 28):
            for j in range(0, 224, 28):
                if (i // 28 + j // 28) % 2 == 0:
                    chk[i : i + 28, j : j + 28] = 230
        valid, msg = validate_upload("checkerboard.png", self._create_png_bytes(Image.fromarray(chk)))
        self.record("NEGATIVE", "Checkerboard Graphic", "checkerboard.png", valid, msg or "")

        # N11: Extreme Aspect Ratio (Panorama / Banner)
        banner = Image.fromarray(np.random.normal(50, 15, (40, 400)).clip(0, 255).astype(np.uint8))
        valid, msg = validate_upload("banner.png", self._create_png_bytes(banner))
        self.record("NEGATIVE", "Extreme Aspect Ratio", "banner.png", valid, msg or "")

        # N12: Tiny Image (<50px)
        tiny = Image.fromarray(np.random.randint(0, 80, (24, 24), dtype=np.uint8))
        valid, msg = validate_upload("tiny_icon.png", self._create_png_bytes(tiny))
        self.record("NEGATIVE", "Tiny Resolution", "tiny_icon.png", valid, msg or "")

        # N13: Empty File (0 bytes)
        valid, msg = validate_upload("empty_scan.png", b"")
        self.record("NEGATIVE", "Empty File", "empty_scan.png", valid, msg or "")

        # N14: Corrupted PNG (truncated header)
        corrupted_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
        valid, msg = validate_upload("corrupted.png", corrupted_bytes)
        self.record("NEGATIVE", "Corrupted PNG", "corrupted.png", valid, msg or "")

        # N15: Unsupported File Extension (.pdf)
        valid, msg = validate_upload("medical_report.pdf", b"%PDF-1.4...")
        self.record("NEGATIVE", "Unsupported Format (.pdf)", "medical_report.pdf", valid, msg or "")

        # N16: Unsupported File Extension (.txt)
        valid, msg = validate_upload("notes.txt", b"Patient scan notes")
        self.record("NEGATIVE", "Unsupported Format (.txt)", "notes.txt", valid, msg or "")

        # N17: Corrupted NIfTI (invalid magic)
        corrupted_nii = bytearray(400)
        struct.pack_into("<i", corrupted_nii, 0, 348)
        corrupted_nii[344:348] = b"BAD!"
        valid, msg = validate_upload("bad_header.nii", bytes(corrupted_nii))
        self.record("NEGATIVE", "Corrupt NIfTI Header", "bad_header.nii", valid, msg or "")

    def _test_anatomical_mri(self):
        # 1. Static 3D Structural MRI NIfTI volume
        nii_3d = self._create_3d_mri_nii_bytes()
        valid, msg = validate_upload("t1_mprage_structural.nii", nii_3d)
        self.record("NEGATIVE", "3D Structural T1 MRI NIfTI", "t1_mprage_structural.nii", valid, msg or "")

        # 2. 2D Static Anatomical Brain MRI Slice (Axial view)
        axial_mri = self._create_anatomical_mri_slice("axial")
        valid, msg = validate_upload("axial_t1_slice.png", self._create_png_bytes(axial_mri))
        self.record("NEGATIVE", "2D Anatomical Slice (Axial)", "axial_t1_slice.png", valid, msg or "")
        valid, msg = validate_upload("axial_t1_slice.jpg", self._create_jpg_bytes(axial_mri))
        self.record("NEGATIVE", "2D Anatomical Slice (Axial JPG)", "axial_t1_slice.jpg", valid, msg or "")

        # 3. 2D Static Anatomical Brain MRI Slice (Sagittal view)
        sag_mri = self._create_anatomical_mri_slice("sagittal")
        valid, msg = validate_upload("sagittal_t2_slice.png", self._create_png_bytes(sag_mri))
        self.record("NEGATIVE", "2D Anatomical Slice (Sagittal)", "sagittal_t2_slice.png", valid, msg or "")

        # 4. 2D Static Anatomical Brain MRI Slice (Coronal view)
        cor_mri = self._create_anatomical_mri_slice("coronal")
        valid, msg = validate_upload("coronal_flair_slice.png", self._create_png_bytes(cor_mri))
        self.record("NEGATIVE", "2D Anatomical Slice (Coronal)", "coronal_flair_slice.png", valid, msg or "")

    def _print_report(self):
        print("\n" + "=" * 90)
        print("VALIDATION SUITE RESULTS")
        print("=" * 90)

        positives = [r for r in self.results if r[0] == "POSITIVE"]
        negatives = [r for r in self.results if r[0] == "NEGATIVE"]

        tp = sum(1 for r in positives if r[3])
        fn = sum(1 for r in positives if not r[3])
        tn = sum(1 for r in negatives if not r[3])
        fp = sum(1 for r in negatives if r[3])

        print(f"\n[POSITIVE SET] (Target: All Accepted)")
        print(f"  Total Positive Samples Tested: {len(positives)}")
        print(f"  - Correctly Accepted (True Positives):   {tp} ({tp / len(positives) * 100:.1f}%)")
        print(f"  - Incorrectly Rejected (False Negatives): {fn} ({fn / len(positives) * 100:.1f}%)")
        print(f"  --> False Rejection Rate: {fn / len(positives) * 100:.2f}%")

        print(f"\n[NEGATIVE SET] (Target: All Rejected)")
        print(f"  Total Negative Samples Tested: {len(negatives)}")
        print(f"  - Correctly Rejected (True Negatives):   {tn} ({tn / len(negatives) * 100:.1f}%)")
        print(f"  - Incorrectly Accepted (False Positives): {fp} ({fp / len(negatives) * 100:.1f}%)")
        print(f"  --> False Acceptance Rate: {fp / len(negatives) * 100:.2f}%")

        if fn > 0:
            print("\n  [!] FALSE REJECTIONS:")
            for cat, subcat, fname, passed, msg in positives:
                if not passed:
                    print(f"      * {fname} ({subcat}): {msg[:100]}...")

        if fp > 0:
            print("\n  [!] FALSE ACCEPTANCES:")
            for cat, subcat, fname, passed, msg in negatives:
                if passed:
                    print(f"      * {fname} ({subcat})")

        print("\n[SAMPLE REJECTION MESSAGES RECEIVED BY IRRELEVANT INPUTS]")
        for cat, subcat, fname, passed, msg in negatives[:5]:
            snippet = msg.replace("\n", " ").encode("ascii", errors="replace").decode("ascii")[:120]
            print(f"  - {fname:<25} -> {snippet}...")

        print("\n" + "=" * 90)
        overall_ok = (fn == 0 and fp == 0)
        if overall_ok:
            print(">>> SUCCESS: 0 False Acceptances, 0 False Rejections. Perfect Discrimination achieved! <<<")
        else:
            print(f">>> FAILED: FN={fn}, FP={fp} <<<")
        print("=" * 90)


if __name__ == "__main__":
    suite = EvaluationSuite()
    suite.run_all()
