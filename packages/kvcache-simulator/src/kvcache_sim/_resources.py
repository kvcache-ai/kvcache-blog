from __future__ import annotations

from importlib import resources
from pathlib import Path
import hashlib
import os
import re
import tempfile

RESOURCE_PACKAGE = "kvcache_sim.resources"


def user_temp_suffix() -> str:
    try:
        return f"-uid{os.getuid()}"
    except AttributeError:
        return ""


def package_resource_path(name: str) -> Path:
    resource = resources.files(RESOURCE_PACKAGE).joinpath(name)
    try:
        path = Path(resource)
        if path.exists():
            return path
    except TypeError:
        pass

    payload = resource.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    target = Path(tempfile.gettempdir()) / f"kvcache-simulator{user_temp_suffix()}-{digest}-{safe_name}"
    if not target.exists() or target.read_bytes() != payload:
        target.write_bytes(payload)
    return target
