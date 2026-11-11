"""Make a reproducible source archive and per-file SHA-256 manifest."""
import argparse
import hashlib
from pathlib import Path
import zipfile


def included(path, root):
    relative = path.relative_to(root)
    if any(part in {".git", "__pycache__", ".venv", "build", "dist", "demo_run", "transformer_run"}
           or part.endswith(".egg-info") for part in relative.parts):
        return False
    return path.is_file() and path.name != "MANIFEST.sha256" and path.suffix not in {".pyc", ".pyo"}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(root.parent/"output/CALMS_SAVANT_Reference_v0.1.0.zip"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output == root or root in output.parents:
        parser.error("output must be outside the source directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if included(p, root))
    manifest = "".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+p.relative_to(root).as_posix()+"\n" for p in files)
    (root/"MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    files.append(root/"MANIFEST.sha256")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            item = zipfile.ZipInfo("CALMS-SAVANT/"+path.relative_to(root).as_posix(), (2026,11,11,0,0,0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            archive.writestr(item, path.read_bytes())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("archive verification failed")
    print(f"{output} | {len(files)} files | {output.stat().st_size} bytes")
    print("sha256", hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
