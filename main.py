"""Entry point.

    python main.py data/participation/2025-09-03.pdf

The task is described by the YAML config next to the PDF, so there is nothing
else to pass.
"""

import sys
from pathlib import Path

from pipeline import runner


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python main.py <pdf_path>")
        raise SystemExit(2)
    # resolve: output/ and logs/ mirror data/, which needs an absolute path
    out = runner.run(Path(sys.argv[1]).resolve())
    if out is None:  # every page failed; any previous CSV was left alone
        print("no rows produced - nothing written. See the run log.")
        raise SystemExit(1)
    print(f"wrote {out.csv}")
    if out.canvas:
        print(f"wrote {out.canvas}  (upload this to Canvas)")
    else:
        print("no Canvas file yet: some sheets are still unidentified. Re-run "
              "this command from a terminal to name them - the reads are "
              "cached, so it costs no model calls.")


if __name__ == "__main__":
    main()
