"""PDF downloader — fetch papers from arXiv or local paths."""
from __future__ import annotations

from pathlib import Path

import requests

from .constants import _ARXIV_PDF_URL, _TIMEOUT, _PDF_DOWNLOAD_DIR

class PDFDownloader:
    """download a PDF from arXiv or a local path."""

    def fetch(self, source: str) -> Path:
        """return a local Path to the PDF for *source*.

        *source* is either a local path or an arXiv ID (e.g. '2407.21783').
        """
        local = Path(source)
        if local.exists():
            return local

        # treat source as arXiv ID
        arxiv_id = source.strip()
        dest = _PDF_DOWNLOAD_DIR / f"{arxiv_id}.pdf"
        if dest.exists():
            return dest

        url = _ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
        print(f"  downloading {url} ...")
        resp = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "EEE-pipeline/1.0"})
        resp.raise_for_status()

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        print(f"  saved to {dest}")
        return dest


