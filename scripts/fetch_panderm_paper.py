"""Retrieve the official paper checkpoint and split metadata without duplicate images."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import requests

FOLDER = ROOT / 'artifacts/panderm_paper'
MODEL_ID, MODEL_SIZE = '1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE', 1418820654
DATA_ID, DATA_SIZE = '1D9Q4B50Z5tyj5fd5EE9QWmFrg66vGvfA', 3245963518


def read_range(identity, start, end):
    for attempt in range(4):
        try:
            response = requests.get('https://drive.usercontent.google.com/download',
                params={'id': identity, 'export': 'download', 'confirm': 't'},
                headers={'Range': f'bytes={start}-{end}'}, timeout=(15, 90))
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != end-start+1:
                raise ValueError('Server did not honor exact byte range')
            if not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                raise ValueError('Returned range does not match request')
            return response.content
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


class RemoteZip(io.RawIOBase):
    def __init__(self, identity, size):
        self.identity, self.size, self.pos = identity, size, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        return self.pos
    def read(self, size=-1):
        if size < 0: size = self.size-self.pos
        if not size: return b''
        end = min(self.pos+size, self.size)-1
        content = read_range(self.identity, self.pos, end)
        self.pos = end+1
        return content


def main():
    from src.models.panderm import sha256_file
    FOLDER.mkdir(parents=True, exist_ok=True)
    model = FOLDER / 'panderm_large.pth'
    if not model.exists():
        chunk_size = 8 * 1024**2
        parts = FOLDER / 'large_chunks'
        parts.mkdir(exist_ok=True)
        count = (MODEL_SIZE+chunk_size-1)//chunk_size
        def download(index):
            start, end = index*chunk_size, min((index+1)*chunk_size, MODEL_SIZE)-1
            path = parts / f'{index:04d}.part'
            if not path.exists() or path.stat().st_size != end-start+1:
                temporary = path.with_suffix('.tmp')
                temporary.write_bytes(read_range(MODEL_ID, start, end))
                temporary.replace(path)
            return index
        with ThreadPoolExecutor(max_workers=24) as pool:
            for n, _ in enumerate(pool.map(download, range(count)), 1):
                if n % 20 == 0 or n == count:
                    print(f'Large checkpoint parts: {n}/{count}', flush=True)
        temporary = model.with_suffix('.download')
        with temporary.open('wb') as stream:
            for index in range(count):
                with (parts / f'{index:04d}.part').open('rb') as source:
                    for block in iter(lambda: source.read(8*1024**2), b''):
                        stream.write(block)
        assert temporary.stat().st_size == MODEL_SIZE
        temporary.replace(model)
    assert model.stat().st_size == MODEL_SIZE
    print('Checkpoint downloaded', sha256_file(model), flush=True)
    if not (FOLDER / 'metadata/HAM10000_clean/ISIC2018_splits/HAM_clean.csv').exists():
        with zipfile.ZipFile(RemoteZip(DATA_ID, DATA_SIZE)) as archive:
            (FOLDER / 'archive_entries.json').write_text(json.dumps([
                {'name': entry.filename, 'bytes': entry.file_size, 'crc': entry.CRC} for entry in archive.infolist()]))
            for entry in archive.infolist():
                if entry.filename.lower().endswith(('.csv', '.txt', '.json')):
                    target = FOLDER / 'metadata' / entry.filename
                    assert target.resolve().is_relative_to((FOLDER / 'metadata').resolve())
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(entry))
    print('Official split metadata available.', flush=True)


if __name__ == '__main__':
    main()
