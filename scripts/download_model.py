"""
download_model.py - Robust, resumable chunked downloader for Hugging Face model weights.
With automatic retry on network timeout.
"""

import os
import sys
import time
import requests
from pathlib import Path
from tqdm import tqdm


def download_file_with_retry(url: str, dest_path: Path, max_retries: int = 10):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"[Downloader] File already exists: {dest_path}")
        return

    temp_path = dest_path.with_suffix(".downloading")

    for attempt in range(1, max_retries + 1):
        existing_bytes = temp_path.stat().st_size if temp_path.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"
            print(f"[Downloader] (Attempt {attempt}/{max_retries}) Resuming {dest_path.name} from byte {existing_bytes / (1024*1024):.1f} MB...")
        else:
            print(f"[Downloader] (Attempt {attempt}/{max_retries}) Downloading {url} -> {dest_path.name}")

        try:
            response = requests.get(url, stream=True, timeout=30, headers=headers)
            
            if response.status_code == 206:
                content_range = response.headers.get("Content-Range", "")
                total_size = int(content_range.split("/")[-1]) if "/" in content_range else existing_bytes + int(response.headers.get("content-length", 0))
                mode = "ab"
            elif response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0))
                existing_bytes = 0
                mode = "wb"
            elif response.status_code == 416:
                # Requested Range Not Satisfiable -> file already complete!
                temp_path.rename(dest_path)
                print(f"[Downloader] File completed (416 Range Full): {dest_path.name}")
                return
            else:
                response.raise_for_status()

            chunk_size = 1024 * 1024  # 1MB chunks

            with open(temp_path, mode) as f, tqdm(
                total=total_size, initial=existing_bytes, unit="B", unit_scale=True, desc=dest_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Completed successfully
            temp_path.rename(dest_path)
            print(f"[Downloader] Successfully saved {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
            return

        except (requests.exceptions.RequestException, Exception) as e:
            print(f"[Downloader] Warning on attempt {attempt}: {e}. Retrying in 3 seconds...")
            time.sleep(3)

    raise RuntimeError(f"Failed to download {dest_path.name} after {max_retries} attempts.")


def main():
    target_dir = Path("models/baseline")
    target_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://huggingface.co/distilbert-base-uncased-distilled-squad/resolve/main"
    files = [
        "config.json",
        "tokenizer_config.json",
        "vocab.txt",
        "tokenizer.json",
        "model.safetensors",
    ]

    for fname in files:
        url = f"{base_url}/{fname}"
        dest = target_dir / fname
        download_file_with_retry(url, dest)

    print("\n[Downloader] All baseline files verified in models/baseline/!")


if __name__ == "__main__":
    main()
