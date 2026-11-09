"""Content revisions shared by training dependencies and model bundle checks."""
import hashlib
from pathlib import Path


def checkpoint_set_revision(paths, namespace):
    digest=hashlib.sha256(str(namespace).encode('utf-8'))
    for path in sorted((Path(item).resolve() for item in paths),key=lambda item:item.name):
        if not path.is_file():raise FileNotFoundError(path)
        digest.update(path.name.encode('utf-8'))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
