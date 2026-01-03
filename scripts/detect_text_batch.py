import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from doctr.models import detection, ocr_predictor

IMAGE_FILE_EXTENSIONS = [".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp"]


# ---------- Dataset & collate ----------

class OCRImageDataset(Dataset):
    def __init__(self, root: Path):
        super().__init__()
        self.files = [
            f for f in root.iterdir()
            if f.suffix.lower() in IMAGE_FILE_EXTENSIONS
        ]
        self.files.sort()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """
        Returns:
            img: HWC uint8 numpy array
            stem: filename stem (used for output name)
        """
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = np.array(img)  # H, W, 3
        return img, path.stem


def collate_fn(batch):
    """
    Batch is list of (img, stem).
    We return:
        imgs: list of HWC numpy arrays (docTR can handle that)
        stems: list of str
    """
    imgs, stems = zip(*batch)
    return list(imgs), list(stems)


# ---------- Helpers for saving ----------

def _pages_to_txt(pages) -> List[str]:
    """Render each page to plain text (one string per page)."""
    texts = []
    for page in pages:
        lines = []
        for block in page.blocks:
            for line in block.lines:
                words = " ".join(word.value for word in line.words)
                lines.append(words)
            lines.append("")
        texts.append("\n".join(lines).strip())
    return texts


def save_batch_outputs(out, stems: List[str], out_format: str, out_dir: Path):
    """
    out: OCRPredictor output for a batch of pages (1 page per image)
    stems: filename stems (same order as batch)
    """
    out_dir.mkdir(exist_ok=True)
    if out_format == "json":
        exported = out.export()  # dict with "pages"
        pages = exported["pages"]
        for page_dict, stem in zip(pages, stems):
            single = {"pages": [page_dict]}
            txt = json.dumps(single, indent=2)
            (out_dir / f"{stem}.json").write_text(txt, encoding="utf-8")

    elif out_format == "txt":
        page_texts = _pages_to_txt(out.pages)
        for text, stem in zip(page_texts, stems):
            (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")

    elif out_format == "xml":
        xml_pages = out.export_as_xml()
        for (xml_bytes, xml_tree), stem in zip(xml_pages, stems):
            xml_tree.write(out_dir / f"{stem}.xml", encoding="utf-8", xml_declaration=True)
    else:
        raise ValueError(f"Unsupported format: {out_format}")


# ---------- Main ----------

def main(args):
    # Choose device
    assert torch.cuda.is_available(), "CUDA not available, but script is set up for GPU use only."
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Build models on CPU first
    det_model = detection.__dict__[args.detection](
        pretrained=True,
        bin_thresh=args.bin_thresh,
        box_thresh=args.box_thresh,
    )

    predictor = ocr_predictor(
        det_model,
        args.recognition,
        pretrained=True,
        reco_bs=args.batch_size,   # batched recognition
        preserve_aspect_ratio=False,
        symmetric_pad=False,
    )

    # Move whole predictor (which contains detection + recognition) to GPU
    predictor = predictor.to(device)

    root = Path(args.path)
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # --- PDFs (simple single-process) ---
    pdf_files = [
        f for f in root.iterdir()
        if f.suffix.lower() == ".pdf"
    ]

    from doctr.io import DocumentFile
    for pdf in tqdm(pdf_files, desc="PDFs"):
        doc = DocumentFile.from_pdf(pdf)
        # predictor handles pre-processing & batching internally
        with torch.no_grad():
            out = predictor(doc)
        if args.format == "json":
            txt = json.dumps(out.export(), indent=2)
            (out_dir / f"{pdf.stem}.json").write_text(txt, encoding="utf-8")
        elif args.format == "txt":
            text = out.render()
            (out_dir / f"{pdf.stem}.txt").write_text(text, encoding="utf-8")
        elif args.format == "xml":
            xml_pages = out.export_as_xml()
            for i, (xml_bytes, xml_tree) in enumerate(xml_pages):
                xml_tree.write(out_dir / f"{pdf.stem}_{i}.xml", encoding="utf-8")

    # --- Images via DataLoader + multiprocessing ---
    dataset = OCRImageDataset(root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,  # multiprocessing workers
        collate_fn=collate_fn,
        pin_memory=True,
    )

    for imgs, stems in tqdm(loader, desc="Images"):
        # imgs is a list of HWC numpy arrays
        # THIS is the format docTR expects for batched processing
        with torch.no_grad():
            out = predictor(imgs)

        save_batch_outputs(out, stems, args.format, out_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DocTR multiprocess batched OCR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("path", type=str, help="Directory with images and/or PDFs")
    parser.add_argument("--detection", type=str, default="fast_base")
    parser.add_argument("--bin-thresh", type=float, default=0.3)
    parser.add_argument("--box-thresh", type=float, default=0.1)
    parser.add_argument("--recognition", type=str, default="crnn_vgg16_bn")
    parser.add_argument(
        "-f", "--format", choices=["txt", "json", "xml"], default="txt", help="Output format"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=24, help="Batch size for images"
    )
    parser.add_argument(
        "-j", "--num-workers", type=int, default=8, help="DataLoader workers (multiprocessing)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
