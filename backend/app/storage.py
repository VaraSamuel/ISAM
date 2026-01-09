import os
from fastapi import UploadFile


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


async def save_upload(file: UploadFile, dest_path: str) -> None:
    """Stream upload to disk (safe for large files)."""
    ensure_dir(os.path.dirname(dest_path))
    with open(dest_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def get_run_dir(data_dir: str, run_id: str) -> str:
    return os.path.join(data_dir, run_id)
