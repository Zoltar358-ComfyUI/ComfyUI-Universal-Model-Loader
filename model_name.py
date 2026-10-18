"""Display names are path-only: no file I/O, hashing or network access."""
import posixpath


def model_name(value, folder=False):
    name = posixpath.basename(str(value).replace("\\", "/").rstrip("/"))
    return name if folder else posixpath.splitext(name)[0]
