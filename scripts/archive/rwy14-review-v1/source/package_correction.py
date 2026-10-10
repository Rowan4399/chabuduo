"""Bundle updated review files; the already delivered real cache is unchanged."""

import hashlib
import json
import zipfile
from pathlib import Path

root = Path("runs/rwy14-review-v1")
files = [
    p
    for p in root.rglob("*")
    if p.is_file() and p.suffix != ".zip" and p.name != "checksums.json"
]
checksums = {
    p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in files
}
checksum_path = root / "checksums.json"
checksum_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
with zipfile.ZipFile(root / "tcvae-correction.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in files + [checksum_path]:
        relative = p.relative_to(root)
        if relative.parts[0] in {"calibration", "calibration-final"}:
            continue
        z.write(p, relative.as_posix())
print(root / "tcvae-correction.zip")
