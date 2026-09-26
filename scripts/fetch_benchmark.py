"""Restore pinned benchmark papers locally; refuse silent corpus changes."""
from __future__ import annotations

import json
from pathlib import Path
import re

import httpx

from scripts.benchmark import BENCHMARK, ROOT, digest, load_benchmark, normalize
from scripts.fetch_corpus import extract_text


def fetch_paper(paper: dict, directory: Path) -> str:
    path = directory / paper["file"]
    if path.exists():
        if digest(normalize(path.read_text(encoding="utf-8"))) != paper["normalized_sha256"]:
            raise ValueError(f"Local corpus differs: {path}. Preserve it and restore the pinned version separately.")
        return "verified"
    match = re.search(re.escape(paper["arxiv_id"]) + r"v\d+", paper["version"])
    if not match:
        raise ValueError(f"Missing arXiv revision: {paper['source_id']}")
    response = httpx.get(f"https://arxiv.org/html/{match.group()}", timeout=90, follow_redirects=True)
    response.raise_for_status()
    text = extract_text(response.text)
    if digest(normalize(text)) != paper["normalized_sha256"]:
        raise ValueError(f"Remote rendering/extraction drift: {paper['source_id']}; downloaded text was NOT saved. Restore an archived local copy matching the manifest.")
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "downloaded"


def main() -> None:
    manifest = json.loads((BENCHMARK / "manifest.json").read_text(encoding="utf-8"))
    for paper in manifest["papers"]:
        print(f"{fetch_paper(paper, ROOT / 'eval/corpus')}: {paper['source_id']}")
    load_benchmark()
    print("Pinned corpus and annotations verified.")


if __name__ == "__main__":
    main()
