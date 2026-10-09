"""Verify and restore the complete data archive using Python's standard library."""
import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 4 * 1024 * 1024


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            result.update(block)
    return result.hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe archive path: {relative}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing files with different content')
    args = parser.parse_args()
    folder = ROOT / 'archive'
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    workspace = ROOT / '.local'
    workspace.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='restore-', dir=workspace) as temp:
        archive_path = Path(temp) / 'full-project-data.zip'
        with archive_path.open('wb') as assembled:
            for part in manifest['parts']:
                path = inside(folder, part['path'])
                if path.stat().st_size != part['size'] or digest(path) != part['sha256']:
                    raise ValueError(f'Part checksum mismatch: {part["path"]}')
                with path.open('rb') as source:
                    shutil.copyfileobj(source, assembled, CHUNK)
        records = manifest['files']
        if len({record['path'] for record in records}) != len(records):
            raise ValueError('Duplicate manifest paths')
        with zipfile.ZipFile(archive_path) as archive:
            if sorted(archive.namelist()) != sorted(record['path'] for record in records):
                raise ValueError('Archive members do not match the manifest')
            # Check conflicts before restoring any files.
            if not args.verify_only and not args.overwrite:
                for record in records:
                    target = inside(ROOT, record['path'])
                    if target.exists() and (not target.is_file() or digest(target) != record['sha256']):
                        raise FileExistsError(f'Existing file differs: {target}. Use --overwrite only if intended.')
            for index, record in enumerate(records, 1):
                target = inside(ROOT, record['path'])
                same = not args.verify_only and target.is_file() and target.stat().st_size == record['size'] and digest(target) == record['sha256']
                restore = not args.verify_only and not same
                staged = Path(temp) / 'current-file'
                checksum = hashlib.sha256()
                count = 0
                with archive.open(record['path']) as source:
                    output = staged.open('wb') if restore else None
                    try:
                        for block in iter(lambda: source.read(CHUNK), b''):
                            checksum.update(block)
                            count += len(block)
                            if output:
                                output.write(block)
                    finally:
                        if output:
                            output.close()
                if count != record['size'] or checksum.hexdigest() != record['sha256']:
                    raise ValueError(f'File checksum mismatch: {record["path"]}')
                if restore:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    staged.replace(target)
                if index % 200 == 0:
                    print(f'Checked {index}/{len(records)} files', flush=True)
        print(f'All {len(records)} archived files verified' + ('.' if args.verify_only else ' and restored.'))


if __name__ == '__main__':
    main()
