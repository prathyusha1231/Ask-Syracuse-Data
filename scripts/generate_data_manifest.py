from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class FileEntry:
    path: str
    size_bytes: int
    modified_utc: str
    sha256: str | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(data_raw_dir: Path, include_hashes: bool) -> dict:
    created_utc = datetime.now(timezone.utc).isoformat()
    entries: list[FileEntry] = []

    try:
        data_root_display = data_raw_dir.relative_to(Path.cwd()).as_posix()
    except ValueError:
        data_root_display = data_raw_dir.as_posix()

    if data_raw_dir.exists():
        for p in sorted(data_raw_dir.rglob("*")):
            if not p.is_file():
                continue
            try:
                rel = p.relative_to(Path.cwd()).as_posix()
            except ValueError:
                rel = p.as_posix()
            stat = p.stat()
            modified_utc = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            digest = _sha256(p) if include_hashes else None
            entries.append(
                FileEntry(
                    path=rel,
                    size_bytes=stat.st_size,
                    modified_utc=modified_utc,
                    sha256=digest,
                )
            )

    return {
        "manifest_version": 1,
        "created_utc": created_utc,
        "data_root": data_root_display,
        "file_count": len(entries),
        "files": [asdict(e) for e in entries],
        "notes": [
            "This manifest captures the exact filenames present under data/raw at generation time.",
            "Use it for reproducibility (data freshness, integrity checks) when running in environments where the raw snapshots are not tracked in Git.",
        ],
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    # Ensure paths are emitted relative to repo root (best-effort)
    try:
        os.chdir(repo_root)
    except Exception:
        pass
    data_raw_dir = repo_root / "data" / "raw"
    out_path = repo_root / "data_manifest.json"

    include_hashes = True
    manifest = build_manifest(data_raw_dir, include_hashes=include_hashes)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({manifest['file_count']} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
