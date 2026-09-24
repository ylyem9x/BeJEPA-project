import fcntl
import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
MANIFEST = DATA_DIR / "data_manifest.json"


def restore_archive(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file() and path.stat().st_size == metadata["bytes"]:
            return path
        temporary = None
        try:
            with tempfile.TemporaryFile(dir=path.parent) as compressed:
                for part in metadata["parts"]:
                    source = path.parent / part["path"]
                    if not source.is_file():
                        raise FileNotFoundError(f"Missing bundled data: {source}. Download the complete repository.")
                    if source.stat().st_size != part["bytes"]:
                        raise ValueError(f"Incomplete data part: {source}")
                    with source.open("rb") as stream:
                        shutil.copyfileobj(stream, compressed, length=1024 * 1024)
                compressed.seek(0)
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
                    temporary = Path(output.name)
                    with gzip.GzipFile(fileobj=compressed) as archive:
                        shutil.copyfileobj(archive, output, length=1024 * 1024)
                if temporary.stat().st_size != metadata["bytes"]:
                    raise ValueError(f"Incomplete restored data: {path.name}")
                os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return path


def ensure_motion_data(path):
    path = Path(path).expanduser().resolve()
    if path.parent == DATA_DIR:
        manifest = json.loads(MANIFEST.read_text())
        if path.name in manifest:
            return restore_archive(path, manifest[path.name])
    if not path.is_file():
        raise FileNotFoundError(f"Motion archive not found: {path}")
    return path


def main():
    for name in json.loads(MANIFEST.read_text()):
        path = ensure_motion_data(DATA_DIR / name)
        print(f"Ready: {path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
