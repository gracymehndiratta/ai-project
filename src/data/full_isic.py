"""Download and validate the complete official ISIC 2018 Task 3 release."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2018/"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
FILES = [
    "ISIC2018_Task3_Training_Input.zip",
    "ISIC2018_Task3_Training_GroundTruth.zip",
    "ISIC2018_Task3_Training_LesionGroupings.csv",
    "ISIC2018_Task3_Validation_Input.zip",
    "ISIC2018_Task3_Validation_GroundTruth.zip",
    "ISIC2018_Task3_Test_Input.zip",
    "ISIC2018_Task3_Test_GroundTruth.zip",
]


def _request(url, **kwargs):
    for attempt in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, **kwargs), timeout=45)
        except OSError:
            if attempt == 5:
                raise
            time.sleep(min(attempt + 1, 5))


def download_file(name: str, archive_dir: Path, workers: int = 32) -> dict:
    """Resume bounded range requests; verify ETag, range, size and ZIP CRCs."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + name
    with _request(url, method="HEAD") as response:
        size = int(response.headers["Content-Length"])
        etag = response.headers["ETag"].strip('"')
    target = archive_dir / name
    receipt = target.with_suffix(target.suffix + ".verified.json")
    spec = {"url": url, "size": size, "etag": etag}
    if target.exists() and target.stat().st_size == size and receipt.exists():
        saved = json.loads(receipt.read_text())
        if all(saved.get(k) == v for k, v in spec.items()):
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest == saved.get("sha256"):
                print(f"Already verified: {name}", flush=True)
                return saved

    chunk_size = 1024 * 1024
    parts = archive_dir / (name + ".parts")
    parts.mkdir(exist_ok=True)
    identity = parts / "source.json"
    if identity.exists() and json.loads(identity.read_text()) != spec:
        raise RuntimeError(f"Remote file changed; preserve/remove old parts before retrying: {parts}")
    identity.write_text(json.dumps(spec))

    def fetch(index):
        start = index * chunk_size
        end = min(start + chunk_size, size) - 1
        path = parts / f"{index:06d}.part"
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(8):
            try:
                with _request(url, headers={"Range": f"bytes={start}-{end}", "If-Match": f'"{etag}"'}) as response:
                    expected_range = f"bytes {start}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError(f"Server did not honor range: {name} {index}")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise RuntimeError(f"Truncated chunk: {name} {index}")
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(data)
                temporary.replace(path)
                return
            except (OSError, RuntimeError):
                if attempt == 7:
                    raise
                time.sleep(min(attempt + 1, 5))

    count = (size + chunk_size - 1) // chunk_size
    print(f"Downloading full {name}: {size / 1e9:.3f} GB, {count} chunks", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, future in enumerate(as_completed(pool.submit(fetch, i) for i in range(count)), 1):
            future.result()
            if done % 128 == 0 or done == count:
                print(f"  {name}: {done}/{count} chunks ({100 * done / count:.1f}%)", flush=True)
    digest = hashlib.sha256()
    md5 = hashlib.md5()
    temporary = target.with_suffix(target.suffix + ".partial")
    with temporary.open("wb") as output:
        for index in range(count):
            data = (parts / f"{index:06d}.part").read_bytes()
            digest.update(data)
            md5.update(data)
            output.write(data)
    if "-" not in etag and md5.hexdigest() != etag:
        raise RuntimeError(f"MD5 checksum mismatch: {name}")
    if name.endswith(".zip"):
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"ZIP CRC validation failed: {bad}")
    temporary.replace(target)
    spec["sha256"] = digest.hexdigest()
    receipt.write_text(json.dumps(spec, indent=2))
    for part in parts.glob("*.part"):
        part.unlink()
    identity.unlink()
    for part in parts.glob("*.tmp"):
        part.unlink()
    parts.rmdir()
    print(f"Verified: {name}", flush=True)
    return spec


def prepare_release(raw_dir: str | Path = "data/raw", workers: int = 32) -> Path:
    raw_dir = Path(raw_dir)
    archives = raw_dir / "isic2018_archives"
    receipts = []
    # Small labels and groupings first, then the complete image archives.
    for name in sorted(FILES, key=lambda x: "Input" in x):
        receipts.append(download_file(name, archives, workers))
    rows = []
    for partition, expected_count in [("Training", 10015), ("Validation", 193), ("Test", 1512)]:
        prefix = f"ISIC2018_Task3_{partition}"
        folder = raw_dir / "images" if partition == "Training" else raw_dir / f"isic_{partition.lower()}_images"
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archives / f"{prefix}_Input.zip") as archive:
            members = [m for m in archive.infolist() if m.filename.lower().endswith(".jpg") and not m.filename.startswith("__MACOSX/")]
            if len(members) != expected_count or len({Path(m.filename).name for m in members}) != expected_count:
                raise ValueError(f"Expected {expected_count} unique images in {prefix}, found {len(members)}")
            for member in members:
                target = folder / Path(member.filename).name
                if target.exists() and target.stat().st_size == member.file_size:
                    continue
                target.write_bytes(archive.read(member))
        with zipfile.ZipFile(archives / f"{prefix}_GroundTruth.zip") as archive:
            csvs = [n for n in archive.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")]
            if len(csvs) != 1:
                raise ValueError(f"Unexpected ground-truth archive: {csvs}")
            with archive.open(csvs[0]) as stream:
                frame = pd.read_csv(stream)
        frame.columns = frame.columns.str.strip()
        columns = [name.upper() for name in CLASS_NAMES]
        labels = frame[columns].to_numpy()
        if len(frame) != expected_count or frame["image"].duplicated().any():
            raise ValueError(f"Incomplete metadata for {prefix}")
        if not np.isin(labels, [0, 1]).all() or not (labels.sum(axis=1) == 1).all():
            raise ValueError(f"Unexpected one-hot labels in {prefix}; do not silently drop rows")
        result = pd.DataFrame({"image_id": frame["image"], "label": np.array(CLASS_NAMES)[labels.argmax(axis=1)]})
        result["source_partition"] = partition.lower()
        result["image_path"] = [str((folder / f"{name}.jpg").resolve()) for name in result.image_id]
        if partition == "Training":
            groups = pd.read_csv(archives / "ISIC2018_Task3_Training_LesionGroupings.csv")
            result = result.merge(groups.rename(columns={"image": "image_id"}), on="image_id", validate="one_to_one")
            if len(result) != expected_count or result.lesion_id.isna().any():
                raise ValueError("Lesion groupings are incomplete")
            processed = Path("data/processed")
            processed.mkdir(parents=True, exist_ok=True)
            result.to_csv(processed / "HAM10000_metadata.csv", index=False)
        else:
            # Official challenge splits do not expose lesion grouping IDs here.
            result["lesion_id"] = "external_" + result.image_id
        rows.append(result)
        print(f"Prepared complete {partition}: {len(result)} images", flush=True)
    all_rows = pd.concat(rows, ignore_index=True)
    if all_rows.image_id.duplicated().any() or not all(Path(p).exists() for p in all_rows.image_path):
        raise ValueError("Release contains duplicate IDs or missing image files")
    output = Path("data/processed/isic2018_full_manifest.csv")
    temporary = output.with_suffix(".tmp")
    all_rows.to_csv(temporary, index=False)
    temporary.replace(output)
    completion = archives / "download_manifest.json"
    temporary = completion.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipts, indent=2))
    temporary.replace(completion)
    return output
