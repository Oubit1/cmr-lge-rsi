"""RSI archive/checkpoint inspection. No service startup or label operations."""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.control_checkpoint import create_checkpoint, digest, recover_rollback, restore_release, source_path
from app.database import Database
from app.rsi_runtime import inheritance, read_policy, write_record


def record_release(root: Path, checkpoint: Path, commit: str, output: Path) -> dict:
    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True).stdout
    old = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    git("merge-base", "--is-ancestor", old["code_commit"], commit)
    paths = git("diff", "--name-only", old["code_commit"], commit).decode().splitlines()
    installed = {}
    for relative in paths:
        target = source_path(root, relative)
        actual = digest(target)
        expected = subprocess.run(["git", "show", f"{commit}:{relative}"],
                                  cwd=root, capture_output=True)
        if expected.returncode == 0:
            if actual != hashlib.sha256(expected.stdout).hexdigest():
                raise ValueError("concurrent change in release file: " + relative)
        elif actual is not None:
            raise ValueError("deleted release file exists: " + relative)
        installed[relative] = actual
    result = {"schema_version": "control-release-v1", "checkpoint": str(checkpoint),
              "commit": git("rev-parse", commit).decode().strip(), "installed": installed}
    write_record(output, result)
    return {"release_manifest": str(output), "files": len(installed), "commit": result["commit"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    generator = sub.add_parser("prepare-generator")
    generator.add_argument("--per-stage",type=int,default=8)
    generator.add_argument("--version",default="reviewed-generator-v1")
    generator.add_argument("--decision-contract",choices=["doctor-disposition-v1"])
    checkpoint = sub.add_parser("checkpoint")
    checkpoint.add_argument("destination", type=Path)
    rollback = sub.add_parser("rollback")
    rollback.add_argument("checkpoint", type=Path)
    rollback.add_argument("--release-manifest", type=Path, required=True)
    rollback.add_argument("--execute", action="store_true")
    recover = sub.add_parser("recover-rollback")
    recover.add_argument("journal", type=Path)
    recover.add_argument("--execute", action="store_true")
    release = sub.add_parser("record-release")
    release.add_argument("checkpoint", type=Path)
    release.add_argument("--commit", default="HEAD")
    release.add_argument("--output", type=Path, required=True)
    quality = sub.add_parser("prepare-quality")
    quality.add_argument("--version", default="reviewed-pairs-v4")
    verification = sub.add_parser("verify-quality")
    verification.add_argument("--output", type=Path, required=True)
    verification.add_argument("--baseline-ref")
    verification.add_argument("--candidate-root", type=Path)
    verification.add_argument("--history-root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-generator":
        from app.rsi_generator_benchmark import prepare
        if not 2 <= args.per_stage <= 32:
            raise ValueError("generator per-stage must be 2..32")
        result=prepare(args.root,args.version,args.per_stage,args.decision_contract)
    elif args.command == "checkpoint":
        result = create_checkpoint(args.root, args.destination)
    elif args.command == "rollback":
        installed = json.loads(args.release_manifest.read_text(encoding="utf-8"))["installed"]
        result = restore_release(args.root, args.checkpoint, installed, execute=args.execute)
    elif args.command == "recover-rollback":
        result = recover_rollback(args.root, args.journal, execute=args.execute)
    elif args.command == "record-release":
        result = record_release(args.root, args.checkpoint, args.commit, args.output)
    elif args.command == "prepare-quality":
        from app.rsi_quality_benchmark import prepare
        result = prepare(args.root, args.version)
    elif args.command == "verify-quality":
        from app.rsi_quality_benchmark import verify
        result = verify(args.root, args.output, baseline_ref=args.baseline_ref,
                        candidate_root=args.candidate_root, history_root=args.history_root)
    else:
        db = Database(args.root / "runtime/workflow.sqlite")
        from app.rsi_quality_benchmark import benchmark_status
        result = {"policy": read_policy(args.root), "inheritance": inheritance(db),
                  "quality_benchmarks": benchmark_status(args.root)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
