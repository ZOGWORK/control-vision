"""Collect the license files of every Python distribution in a PyInstaller bundle.

control_vision.spec calls write_license_bundle() with the module and file names
PyInstaller collected. Each distribution that supplied one of them gets its own
folder of license, notice and copying files copied from the installed package,
and INDEX.txt lists name, version, declared license and copied files. A
distribution that ships no license file is still listed with its declared
license so the gap is visible rather than silent.
"""
from __future__ import annotations

import importlib.metadata as metadata
import re
import shutil
import sys
from pathlib import Path, PurePath
from typing import Iterable

INDEX_NAME = "INDEX.txt"
_LICENSE_FILE = re.compile(r"(licen[cs]e|copying|notice|copyright|third[-_ ]?party)", re.IGNORECASE)
_SOURCE_SUFFIXES = {".py", ".pyc", ".pyi", ".pyd", ".dll", ".so", ".h", ".hpp", ".c", ".cpp"}
# Bundled-file prefixes that belong to no Python distribution.
_NON_DISTRIBUTION = ("base_library", "pyi_rth_", "pyimod", "pyiboot")


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def top_level_names(collected: Iterable[str]) -> set[str]:
    """First import component of each collected module or bundle path."""
    names = set()
    for entry in collected:
        first = re.split(r"[./\\]", str(entry), maxsplit=1)[0]
        if first and not any(first.startswith(prefix) for prefix in _NON_DISTRIBUTION):
            names.add(first)
    return names


def bundled_distributions(collected: Iterable[str], always: Iterable[str] = ()) -> dict[str, metadata.Distribution]:
    """Installed distributions that provide any collected top-level module."""
    providers = metadata.packages_distributions()
    wanted = {canonical_name(name) for name in always}
    for top in top_level_names(collected):
        wanted.update(canonical_name(name) for name in providers.get(top, []))
    found: dict[str, metadata.Distribution] = {}
    for name in sorted(wanted):
        try:
            found[name] = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def license_files(dist: metadata.Distribution) -> list[tuple[PurePath, Path]]:
    """(recorded path, file on disk) for each license-like file the package installed.

    Files in the .dist-info folder are always considered; files inside the
    package itself only at its top level (for example cv2/LICENSE-3RD-PARTY.txt
    or onnxruntime/ThirdPartyNotices.txt) so test data is not swept in.
    """
    result = []
    for recorded in dist.files or []:
        parts = recorded.parts
        name = parts[-1]
        if not _LICENSE_FILE.search(name) or PurePath(name).suffix.lower() in _SOURCE_SUFFIXES:
            continue
        if not (parts[0].endswith(".dist-info") or len(parts) <= 2):
            continue
        located = Path(str(dist.locate_file(recorded)))
        if located.is_file():
            result.append((PurePath(recorded), located))
    return sorted(result, key=lambda item: str(item[0]))


def declared_license(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    expression = meta.get("License-Expression")
    if expression:
        return expression
    declared = (meta.get("License") or "").strip()
    if declared and "\n" not in declared and len(declared) <= 120:
        return declared
    classifiers = [value.split("::")[-1].strip() for value in meta.get_all("Classifier") or []
                   if value.startswith("License ::")]
    if classifiers:
        return "; ".join(classifiers)
    return "see copied license files" if declared else "not declared in package metadata"


def write_license_bundle(collected: Iterable[str], out_dir: Path, always: Iterable[str] = ()) -> list[tuple[str, str]]:
    """Write license folders plus INDEX.txt into out_dir.

    Returns (source file, folder relative to out_dir) pairs in the shape
    PyInstaller's datas expect. out_dir is replaced so files from an earlier
    build cannot linger.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    lines = ["Python distributions bundled in this build and their license files.",
             "Generated at build time from the installed package metadata.", ""]
    runtime_license = Path(sys.base_prefix) / "LICENSE.txt"
    python_dir = out_dir / f"python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if runtime_license.is_file():
        python_dir.mkdir()
        shutil.copyfile(runtime_license, python_dir / "LICENSE.txt")
        lines += [f"Python {sys.version.split()[0]} (runtime): PSF-2.0 and bundled components",
                  f"  {python_dir.name}/LICENSE.txt", ""]
    for name, dist in bundled_distributions(collected, always).items():
        folder = out_dir / f"{name}-{dist.version}"
        files = license_files(dist)
        lines.append(f"{dist.metadata['Name']} {dist.version}: {declared_license(dist)}")
        if files:
            folder.mkdir()
            for recorded, located in files:
                # Drop the .dist-info folder name; keep any package folder
                # (cv2/LICENSE.txt) so same-named files cannot collide.
                parts = recorded.parts[1:] if recorded.parts[0].endswith(".dist-info") else recorded.parts
                target = folder.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(located, target)
                lines.append(f"  {target.relative_to(out_dir).as_posix()}")
        else:
            homepage = dist.metadata.get("Home-page") or ""
            lines.append("  No license file is shipped in this package; terms are as declared above"
                         + (f" ({homepage})." if homepage else "."))
        lines.append("")
    (out_dir / INDEX_NAME).write_text("\n".join(lines), encoding="utf-8")
    return [(str(path), path.parent.relative_to(out_dir).as_posix())
            for path in sorted(out_dir.rglob("*")) if path.is_file()]


if __name__ == "__main__":
    # Manual check: list what would be bundled for the given top-level names.
    for dist_name, dist in bundled_distributions(sys.argv[1:]).items():
        print(dist_name, dist.version, declared_license(dist), len(license_files(dist)))
