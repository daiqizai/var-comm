"""Linux original-runtime byte verification with a bounded directory-fd cache.

No model imports, shell commands, global cache, configuration hooks or writes.
Every invocation reads every admitted file in full and closes its own descriptors.
"""
from collections import OrderedDict
import hashlib
import os
from pathlib import PurePosixPath
import re
import stat
import sys


MAX_DIRECTORY_FDS = 256
READ_BYTES = 4 * 1024 * 1024
_SHA = re.compile(r'[0-9a-f]{64}')


def _need(ok, message):
    if not ok:
        raise RuntimeError(message)


def _absolute(value):
    _need(isinstance(value, str) and value.startswith('/') and
          not value.startswith('//') and '\\' not in value and '\0' not in value,
          'Canonical absolute POSIX path required')
    path = PurePosixPath(value)
    _need(str(path) == value and '..' not in path.parts,
          'Non-canonical or parent-traversing path')
    return path


def validate_runtime_rows(rows, base):
    """Pure lexical/schema checks and path sorting; no filesystem access."""
    base_path = _absolute(os.fspath(base))
    _need(str(base_path) != '/', 'A bounded personal base is required')
    _need(isinstance(rows, list) and 1 <= len(rows) <= 200000,
          'Finite nonempty runtime rows required')
    result = []
    seen = set()
    for row in rows:
        _need(isinstance(row, dict), 'Runtime row must be a mapping')
        path = _absolute(row['path'])
        _need(path != base_path and path.is_relative_to(base_path),
              'Runtime leaf is outside personal base')
        _need(row['path'] not in seen, 'Duplicate runtime leaf')
        _need(isinstance(row['sha256'], str) and _SHA.fullmatch(row['sha256']) and
              type(row['bytes']) is int and row['bytes'] >= 0, 'Invalid byte pin')
        seen.add(row['path'])
        result.append(dict(path=str(path), sha256=row['sha256'], bytes=row['bytes']))
    # Restoration order is by content hash. Group nearby paths for fd reuse;
    # this only orders byte validation, never model inputs or scientific calls.
    result.sort(key=lambda row: row['path'])
    return result, base_path


def _identity(value):
    return value.st_dev, value.st_ino


def _file_state(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


class _Directories:
    def __init__(self):
        self.fds = OrderedDict()
        self.identities = {}
        self.peak = 0
        self.flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

    def close(self):
        while self.fds:
            _, fd = self.fds.popitem(last=False)
            os.close(fd)

    def _room(self, protected=None):
        if len(self.fds) >= MAX_DIRECTORY_FDS:
            name = next(name for name in self.fds if name != protected)
            os.close(self.fds.pop(name))

    def _store(self, path, fd):
        try:
            value = os.fstat(fd)
            _need(stat.S_ISDIR(value.st_mode), 'Opened path is not a directory')
            identity = _identity(value)
            _need(path not in self.identities or self.identities[path] == identity,
                  'Directory identity changed while reopening: ' + str(path))
            self.identities[path] = identity
            self.fds[path] = fd
            self.peak = max(self.peak, len(self.fds))
        except BaseException:
            os.close(fd)
            raise

    def get(self, path):
        missing = []
        current = path
        while current not in self.fds:
            if current == PurePosixPath('/'):
                self._room()
                self._store(current, os.open('/', self.flags))
                break
            missing.append(current)
            current = current.parent
        self.fds.move_to_end(current)
        for target in reversed(missing):
            self._room(protected=current)
            fd = os.open(target.name, self.flags, dir_fd=self.fds[current])
            self._store(target, fd)
            current = target
        return self.fds[path]

    def recheck_namespace(self):
        # Old cached fds may refer to directories renamed during a file read.
        # Start each final path walk at a fresh root fd, never an old anchor.
        # At most two directory fds are open in these unique-directory checks.
        self.close()
        for path in self.identities:
            fd = os.open('/', self.flags)
            current = PurePosixPath('/')
            try:
                _need(_identity(os.fstat(fd)) == self.identities[current],
                      'Root directory identity changed')
                for part in path.parts[1:]:
                    child = os.open(part, self.flags, dir_fd=fd)
                    os.close(fd)
                    fd = child
                    current = current / part
                    _need(_identity(os.fstat(fd)) == self.identities[current],
                          'Directory namespace changed: ' + str(current))
            finally:
                os.close(fd)


def verify_runtime_rows(rows, base):
    """Verify all supplied SHA/size pins, then recheck all used directories.

    `rows` carries path/sha256/bytes (additional provenance fields are untouched).
    Failure raises and closes all fds. This is a byte gate, not model qualification.
    """
    checked, base_path = validate_runtime_rows(rows, base)
    _need(sys.platform.startswith('linux'), 'Runtime byte verifier requires Linux')
    directories = _Directories()
    total = 0
    try:
        directories.get(base_path)  # Includes every ancestor, without symlinks.
        for row in checked:
            path = PurePosixPath(row['path'])
            parent = directories.get(path.parent)
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
            fd = os.open(path.name, flags, dir_fd=parent)
            try:
                before = os.fstat(fd)
                _need(stat.S_ISREG(before.st_mode), 'Runtime leaf is not regular: ' + str(path))
                _need(before.st_size == row['bytes'], 'Runtime leaf size mismatch: ' + str(path))
                digest = hashlib.sha256()
                count = 0
                while True:
                    chunk = os.read(fd, READ_BYTES)
                    if not chunk:
                        break
                    count += len(chunk)
                    _need(count <= row['bytes'], 'Runtime leaf grew while reading: ' + str(path))
                    digest.update(chunk)
                after = os.fstat(fd)
                entry = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                _need(_file_state(before) == _file_state(after) == _file_state(entry),
                      'Runtime leaf changed or was replaced while reading: ' + str(path))
                _need(count == row['bytes'] and digest.hexdigest() == row['sha256'],
                      'Runtime leaf SHA/size mismatch: ' + str(path))
                total += count
            finally:
                os.close(fd)
        directories.recheck_namespace()
        return dict(schema='LEO_RUNTIME_ROWS_BYTE_VERIFICATION_V1', verified_files=len(checked),
                    verified_bytes=total, directories_reverified=len(directories.identities),
                    max_cached_directory_fds=directories.peak, directory_fd_limit=MAX_DIRECTORY_FDS,
                    all_file_bytes_hashed=True, numerical_qualification=False)
    finally:
        directories.close()
