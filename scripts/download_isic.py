#!/usr/bin/env python3
"""
Robust ISIC HAM10000 downloader with resume support and auto-retry.
Uses only Python stdlib — no extra dependencies.
"""

import os
import sys
import time
import ssl
import urllib.request
import urllib.error
import http.client

# --- SSL context: use certifi if available, otherwise skip verification ---
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl._create_unverified_context()

BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2018"

FILES = [
    ("ISIC2018_Task3_Training_Input.zip",         2_600_000_000),  # ~2.6 GB
    ("ISIC2018_Task3_Training_GroundTruth.zip",        36_000),
    ("ISIC2018_Task3_Training_LesionGroupings.csv",   481_000),
    ("ISIC2018_Task3_Validation_Input.zip",        51_000_000),
    ("ISIC2018_Task3_Validation_GroundTruth.zip",       7_000),
    ("ISIC2018_Task3_Test_Input.zip",             401_000_000),
    ("ISIC2018_Task3_Test_GroundTruth.zip",            11_000),
]

DEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CHUNK_SIZE = 1024 * 512   # 512 KB per chunk
MAX_RETRIES = 9999        # retry forever until done
RETRY_DELAY = 20          # seconds between retries


def fmt_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download_file(filename, expected_size):
    url = f"{BASE_URL}/{filename}"
    dest = os.path.join(DEST_DIR, filename)
    os.makedirs(DEST_DIR, exist_ok=True)

    existing = os.path.getsize(dest) if os.path.exists(dest) else 0

    if existing >= expected_size * 0.99:
        print(f"  ✅  {filename} already complete ({fmt_size(existing)}), skipping.")
        return

    attempt = 0
    while True:
        attempt += 1
        existing = os.path.getsize(dest) if os.path.exists(dest) else 0
        try:
            req = urllib.request.Request(url)
            if existing > 0:
                req.add_header("Range", f"bytes={existing}-")
                print(f"  ↩️  Resuming {filename} from {fmt_size(existing)} (attempt {attempt})")
            else:
                print(f"  ⬇️  Downloading {filename} (attempt {attempt})")

            with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as resp:
                mode = "ab" if existing > 0 else "wb"
                downloaded = existing
                t0 = time.time()
                with open(dest, mode) as f:
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        elapsed = time.time() - t0 or 0.001
                        speed = (downloaded - existing) / elapsed
                        print(
                            f"\r    {fmt_size(downloaded)} downloaded  "
                            f"({fmt_size(speed)}/s)",
                            end="", flush=True
                        )
            print(f"\n  ✅  {filename} complete ({fmt_size(downloaded)})")
            return

        except (urllib.error.URLError, http.client.HTTPException,
                ConnectionResetError, TimeoutError, OSError) as e:
            print(f"\n  ⚠️  Error on attempt {attempt}: {e}")
            if attempt >= MAX_RETRIES:
                print(f"  ❌  Giving up on {filename} after {attempt} attempts.")
                return
            print(f"  🔁  Retrying in {RETRY_DELAY}s …")
            time.sleep(RETRY_DELAY)


def main():
    print(f"Saving files to: {os.path.abspath(DEST_DIR)}\n")
    for filename, expected_size in FILES:
        download_file(filename, expected_size)
    print("\n🎉  All downloads finished!")


if __name__ == "__main__":
    main()
