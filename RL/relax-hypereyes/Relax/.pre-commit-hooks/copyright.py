# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import datetime
import os
import re
import sys


COPYRIGHT = """Copyright (c) {year} Relax Authors. All Rights Reserved."""
RE_COPYRIGHT = re.compile(r".*Copyright \(c\) \d{4} Relax", re.IGNORECASE)
RE_INSPIRED = re.compile(r".*This code is .*", re.IGNORECASE)
RE_SRC_LINK = re.compile(r"https:.*\.py", re.IGNORECASE)


def _generate_copyright(comment_mark):
    year = datetime.datetime.now().year
    copyright = COPYRIGHT.format(year=year)

    return [
        (f"{comment_mark} {line}{os.linesep}" if line else f"{comment_mark}{os.linesep}")
        for line in copyright.splitlines()
    ]


def _copy_inspired_statement(original_copyright, comment_mark):
    # only search top max_scan_lines for inspired statments
    max_scan_lines = 10
    head = original_copyright[0:max_scan_lines]
    statements, link_start_idx = [], max_scan_lines
    for i, line in enumerate(head):
        if RE_INSPIRED.search(line) is not None:
            statements.append(line)
            link_start_idx = i
        elif i > link_start_idx:
            if RE_SRC_LINK.search(line) or RE_INSPIRED.search(line):
                statements.append(line)
            else:
                break
    return statements


def _get_comment_mark(path):
    lang_type = re.compile(r"\.(py|pyi|sh)$")
    if lang_type.search(path) is not None:
        return "#"

    lang_type = re.compile(r"\.(h|c|hpp|cc|cpp|cu|go|cuh|proto)$")
    if lang_type.search(path) is not None:
        return "//"

    return None


RE_ENCODE = re.compile(r"^[ \t\v]*#.*?coding[:=]", re.IGNORECASE)
RE_SHEBANG = re.compile(r"^[ \t\v]*#[ \t]?\!")


def _check_copyright(path):
    head = []
    max_scan_lines = 4
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i in range(max_scan_lines):
                head.append(next(f))
    except StopIteration:
        pass

    for idx, line in enumerate(head):
        if RE_COPYRIGHT.search(line) is not None:
            return True

    return False


def _remove_other_copyright(original_contents, comment_mark):
    # Suppose that length of copyright will not be greater than max_scan_lines
    max_scan_lines = min(100, len(original_contents))
    pattern = comment_mark + " Copyright"
    start, end = None, max_scan_lines
    for i, line in enumerate(original_contents[:max_scan_lines]):
        if line.startswith(pattern) and start is None:
            start = i
        if start is not None and not line.startswith(comment_mark):
            end = i
            break
    if start is not None and end is not None and start < end:
        del original_contents[start:end]


def generate_copyright(path, comment_mark):
    original_contents = open(path, "r", encoding="utf-8").readlines()
    head = original_contents[0:4]

    insert_line_no = 0
    for i, line in enumerate(head):
        if RE_ENCODE.search(line) or RE_SHEBANG.search(line):
            insert_line_no = i + 1

    copyright = _generate_copyright(comment_mark)
    inspired_stmt = _copy_inspired_statement(original_contents, comment_mark)
    if inspired_stmt:
        copyright.append(comment_mark + os.linesep)
        copyright.extend(inspired_stmt)

    # Automatically remove other platform copyright while
    # referring their source code.
    _remove_other_copyright(original_contents, comment_mark)

    if insert_line_no == 0:
        new_contents = copyright
        if len(original_contents) > 0 and len(original_contents[0].strip()) != 0:
            new_contents.append(os.linesep)
        new_contents.extend(original_contents)
    else:
        new_contents = original_contents[0:insert_line_no]
        new_contents.append(os.linesep)
        new_contents.extend(copyright)
        if len(original_contents) > insert_line_no and len(original_contents[insert_line_no].strip()) != 0:
            new_contents.append(os.linesep)
        new_contents.extend(original_contents[insert_line_no:])
    new_contents = "".join(new_contents)

    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write(new_contents)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Checker for copyright declaration.")
    parser.add_argument("filenames", nargs="*", help="Filenames to check")
    args = parser.parse_args(argv)

    for path in args.filenames:
        comment_mark = _get_comment_mark(path)
        if comment_mark is None:
            print("warning:Unsupported file", path, file=sys.stderr)
            continue

        if _check_copyright(path):
            continue
        generate_copyright(path, comment_mark)


if __name__ == "__main__":
    sys.exit(main())
