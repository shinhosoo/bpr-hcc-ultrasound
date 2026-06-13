#!/usr/bin/env python3
"""
data/**/pkl/*.pkl 안의 'img_root' 경로 prefix를 현재 프로젝트 루트로 갈아끼운다.

다른 머신에서 만든 pkl을 그대로 옮겨오면 절대경로가 깨져서
DiffMIC / DiffMICv2 학습이 FileNotFoundError로 죽는다 (MedViT는 imagefolder 직접 사용이라 영향 없음).

Usage:
    python3 tools/relocate_pkl_paths.py

    python3 tools/relocate_pkl_paths.py --old /home/user/Documents/test02

    python3 tools/relocate_pkl_paths.py --new /home/hosoo/Documents/test04

    python3 tools/relocate_pkl_paths.py --dry-run
"""
import argparse, os, glob, pickle, shutil, sys


def patch_one(pkl_path: str, new_root: str, old_root: str | None,
              dry_run: bool, verbose: bool) -> tuple[int, int, str]:
    """Return (num_changed, num_total, status)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        return 0, 0, f"skip (not a list): {type(data).__name__}"

    changed = 0
    total = len(data)
    sample_before = None
    sample_after = None

    for rec in data:
        if not isinstance(rec, dict) or "img_root" not in rec:
            continue
        p = rec["img_root"]
        if os.path.exists(p):
            continue
        new_p = None
        if old_root and p.startswith(old_root):
            new_p = new_root + p[len(old_root):]
        else:
            idx = p.find("/data/")
            if idx >= 0:
                new_p = os.path.join(new_root, p[idx + 1:])
        if new_p is None:
            continue
        if not os.path.exists(new_p):
            continue
        if sample_before is None:
            sample_before, sample_after = p, new_p
        rec["img_root"] = new_p
        changed += 1

    if changed == 0:
        return 0, total, "no change needed"

    if not dry_run:
        bak = pkl_path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(pkl_path, bak)
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

    msg = f"{changed}/{total} paths rewritten"
    if verbose and sample_before:
        msg += f"\n      e.g.  {sample_before}\n         -> {sample_after}"
    return changed, total, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="project root (default: parent of tools/)")
    ap.add_argument("--old", default=None,
                    help="old prefix to strip (default: auto-detect from pkl)")
    ap.add_argument("--new", default=None,
                    help="new prefix (default: --root)")
    ap.add_argument("--pkl-glob", default="data/**/pkl/*.pkl",
                    help="glob (relative to root) — default: data/**/pkl/*.pkl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    root = os.path.abspath(a.root or os.path.join(os.path.dirname(__file__), ".."))
    new_root = a.new or root
    old_root = a.old
    print(f"root        = {root}")
    print(f"new prefix  = {new_root}")
    print(f"old prefix  = {old_root or '(auto)'}")
    print(f"glob        = {a.pkl_glob}")
    print(f"dry-run     = {a.dry_run}")
    print()

    pkls = sorted(glob.glob(os.path.join(root, a.pkl_glob), recursive=True))
    if not pkls:
        print("no pkl files found.")
        sys.exit(1)

    grand_changed = grand_total = 0
    for p in pkls:
        changed, total, msg = patch_one(p, new_root, old_root, a.dry_run, a.verbose)
        rel = os.path.relpath(p, root)
        print(f"  {rel}: {msg}")
        grand_changed += changed
        grand_total += total

    print()
    print(f"DONE  rewrote {grand_changed}/{grand_total} records across {len(pkls)} pkls"
          + ("  [dry-run]" if a.dry_run else ""))
    if not a.dry_run and grand_changed:
        print("       .bak backups written next to each modified pkl.")


if __name__ == "__main__":
    main()
