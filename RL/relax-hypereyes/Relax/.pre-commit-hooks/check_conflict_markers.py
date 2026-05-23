# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check for git merge conflict markers, including extended markers (8+ chars).

The built-in ``check-merge-conflict`` hook only detects standard 7-character
markers (``<<<<<<<``, ``=======``, ``>>>>>>>``).  When git lengthens the markers
(e.g. due to nested conflicts or ``diff3`` / ``zdiff3`` conflict style), the
standard hook silently misses them.  This script catches **any** marker length
≥ 7 so that no conflict artifacts slip through.
"""

import argparse
import re
from typing import Sequence


# Match lines that start with 7 or more <, =, or > characters.
CONFLICT_RE = re.compile(rb"^(<{7,}|={7,}|>{7,})(\s|$)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args(argv)

    retcode = 0
    for filename in args.filenames:
        try:
            with open(filename, "rb") as f:
                for lineno, line in enumerate(f, start=1):
                    if CONFLICT_RE.match(line):
                        marker = line.split(b" ")[0].split(b"\t")[0].strip().decode()
                        print(f"{filename}:{lineno}: Conflict marker {marker!r} found")
                        retcode = 1
        except (OSError, UnicodeDecodeError):
            # Skip files that can't be read (binary, permissions, etc.)
            continue

    return retcode


if __name__ == "__main__":
    raise SystemExit(main())
