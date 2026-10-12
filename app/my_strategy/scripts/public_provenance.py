"""Redact host paths from public build metadata without altering native identity."""
from __future__ import annotations

import copy
from pathlib import PurePosixPath, PureWindowsPath
import re


def _absolute(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def public_provenance(value: dict) -> dict:
    """Copy a CZSC source manifest or wheel index, changing only build paths.

    Source files, source tree, compiler options, wheel tags and physical wheel
    hashes remain intact. The original build records stay in the source tree.
    """
    result = copy.deepcopy(value)
    records = [result, *result.get("wheels", [])]
    for record in records:
        build = record.get("build")
        if not isinstance(build, dict):
            continue
        changed = False
        target = build.get("cargo_target_dir")
        if isinstance(target, str) and _absolute(target):
            build["cargo_target_dir"] = "<build-local-cargo-target-dir>"
            changed = True
        flags = build.get("rustflags")
        if isinstance(flags, str):
            def replace(match):
                nonlocal changed
                option, path = match.groups()
                if not _absolute(path.strip("\"'")):
                    return match.group(0)
                changed = True
                replacement = "<build-local-native-library-dir>" if option == "-Lnative=" else "<build-local-dlltool>"
                return option + replacement
            build["rustflags"] = re.sub(r"(-Lnative=|dlltool=)(\"[^\"]*\"|'[^']*'|\S+)", replace, flags)
        if changed:
            build["public_build_paths_redacted"] = True
    return result
