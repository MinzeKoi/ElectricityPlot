import hashlib
import json
from pathlib import Path
import sys
import zlib
import zipfile

sys.path.insert(0, 'tmp/refit_archive_tools')
import py7zr

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

root = Path('data/raw/plegma_20261022')
archive_path = root / 'PlegmaDataset_Clean.7z'
destination = root / 'extracted'
if destination.exists():
    raise ValueError('Extraction destination already exists')
with py7zr.SevenZipFile(archive_path) as archive:
    members = archive.list()
    for member in members:
        path = (destination / member.filename).resolve()
        path.relative_to(destination.resolve())
        if member.is_symlink or ':' in member.filename or '\\' in member.filename:
            raise ValueError('Unsafe archive member')
    destination.mkdir()
    print('Extracting Plegma', flush=True)
    archive.extractall(destination)
inventory = []
for member in members:
    if member.is_directory:
        continue
    path = destination / member.filename
    checksum, fingerprint = 0, hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            checksum = zlib.crc32(block, checksum)
            fingerprint.update(block)
    if path.stat().st_size != member.uncompressed or checksum & 0xffffffff != member.crc32:
        raise ValueError(f'Plegma member mismatch: {member.filename}')
    inventory.append({'path': member.filename, 'bytes': member.uncompressed,
                      'sha256': fingerprint.hexdigest(), 'archive_crc32_verified': True})
data = {'source_url': 'https://pure.strath.ac.uk/ws/portalfiles/portal/180681715/PlegmaDataset_Clean.7z',
        'license': 'CC-BY-4.0', 'archive_bytes': archive_path.stat().st_size,
        'archive_sha256': sha(archive_path), 'kind': 'complete_public_archive',
        'status': 'extracted_members_crc_verified_not_normalized',
        'publisher_checksum_verified': False, 'files': inventory}
(root / 'inventory.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
print('Plegma verified files:', len(inventory), flush=True)
ideal = Path('data/raw/ideal_metadata_20261022')
items = []
for path in ideal.glob('*.zip'):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('IDEAL archive corrupted')
        items.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path),
                      'zip_crc_verified': True, 'members': archive.namelist()})
(ideal / 'inventory.json').write_text(json.dumps({'source_url': 'https://doi.org/10.7488/ds/2836',
    'license': 'CC-BY-4.0', 'scope': 'documentation_and_metadata_only_no_power_measurements',
    'files': items}, indent=2) + '\n', encoding='utf-8')
print('IDEAL verified metadata archives:', len(items), flush=True)
