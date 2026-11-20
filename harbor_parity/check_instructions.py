"""Check that the original-side prompts are byte-identical to the Harbor tasks.

Usage: python harbor_parity/check_instructions.py <harbor-dataset-dir>
where <harbor-dataset-dir> is the output of the adapter (one directory per task,
named by the upstream problem UUID).
"""

import sys
from pathlib import Path

from run_parity import build_instruction, load_problems


def main() -> None:
    dataset_dir = Path(sys.argv[1])
    rows = load_problems(None)
    mismatches = [
        r["id"]
        for r in rows
        if (dataset_dir / r["id"].lower() / "instruction.md").read_text()
        != build_instruction(r["problem"])
    ]
    print(f"checked {len(rows)} problems; mismatches: {len(mismatches)}")
    for pid in mismatches[:10]:
        print("  mismatch:", pid)
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
