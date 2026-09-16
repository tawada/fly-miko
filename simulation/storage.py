"""Atomic saves with bounded retries for busy files on shared filesystems."""
from contextlib import contextmanager, suppress
import errno
import os
from pathlib import Path
import tempfile
import time

RETRYABLE = {errno.EACCES, errno.EPERM, errno.EBUSY}


def replace_with_retry(source, destination, attempts=10):
    """Never delete or truncate the destination to work around a failed rename."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if error.errno not in RETRYABLE or attempt == attempts - 1:
                raise
            time.sleep(min(.025 * 2 ** attempt, .25))


@contextmanager
def atomic_output(path):
    path = Path(path)
    # A distinct sibling prevents stale .tmp ownership and temporary-name clashes.
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        try:
            replace_with_retry(temporary, path)
        except OSError as error:
            raise OSError(error.errno,
                          "Atomic save failed after bounded retries; the previous file was kept. "
                          "Check directory ownership, file locks, and shared-filesystem access",
                          str(path)) from error
    finally:
        # A cleanup failure must not hide the original write/replace error.
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def check_output_directory(directory):
    """Exercise create, write, and replacement without altering saved results."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".storage-check.", dir=directory)
    target = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(b"old")
        with atomic_output(target) as stream:
            stream.write(b"new")
        if target.read_bytes() != b"new":
            raise OSError("Storage replacement verification failed")
    finally:
        target.unlink(missing_ok=True)
