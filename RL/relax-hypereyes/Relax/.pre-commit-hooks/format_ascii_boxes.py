#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Format ASCII box-drawing diagrams in markdown files.

Vertically aligns right-edge characters (┐ ┘ │ ┤) within each
box-drawing diagram found inside fenced code blocks.  CJK / fullwidth
characters are handled correctly (they occupy 2 display columns).

Usage:
    python format_ascii_boxes.py [--check] file1.md file2.md ...

As a pre-commit hook, exits with code 1 if any file was modified.
"""

import argparse
import unicodedata


# ── display-width helpers ────────────────────────────────────────────


def _char_width(ch):
    """Return the display width of *ch* (1 or 2)."""
    eaw = unicodedata.east_asian_width(ch)
    return 2 if eaw in ("W", "F") else 1


def _display_width(s):
    """Return the total display width of string *s*."""
    return sum(_char_width(c) for c in s)


def _rstrip_display(s):
    """Strip trailing ASCII spaces and return ``(stripped, display_width)``."""
    stripped = s.rstrip(" ")
    return stripped, _display_width(stripped)


# ── structural helpers ───────────────────────────────────────────────


def _find_code_blocks(lines):
    """Return ``(start, end)`` ranges of code-block content that contain box
    drawings."""
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("```"):
            content_start = i + 1
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                i += 1
            content_end = i  # exclusive
            if any("┌" in lines[j] for j in range(content_start, min(content_end, len(lines)))):
                blocks.append((content_start, content_end))
        i += 1
    return blocks


# Characters that can appear at a box's left column between ┌ and └.
_LEFT_CONTINUATION = frozenset("│├┤┼")


def _find_boxes(lines):
    """Match each ``┌`` with the nearest ``└`` at the same column.

    Returns a list of ``(start_line, end_line, left_col)`` tuples.
    """
    boxes = []
    for i, line in enumerate(lines):
        for j, ch in enumerate(line):
            if ch != "┌":
                continue
            for k in range(i + 1, len(lines)):
                if j >= len(lines[k]):
                    break
                c = lines[k][j]
                if c == "└":
                    boxes.append((i, k, j))
                    break
                if c not in _LEFT_CONTINUATION:
                    break
    return boxes


def _box_contains(outer, inner):
    os_, oe, ol = outer
    is_, ie, il = inner
    return os_ <= is_ and ie <= oe and ol <= il and (os_, oe, ol) != (is_, ie, il)


def _nesting_depth(box, all_boxes):
    return sum(1 for o in all_boxes if _box_contains(o, box))


def _first_char(line, left_col, target):
    """Return index of the first *target* char right of *left_col*, or
    ``-1``."""
    for j in range(left_col + 1, len(line)):
        if line[j] == target:
            return j
    return -1


def _best_candidate(candidates, expected):
    """Pick the candidate closest to *expected*; prefer ``>= expected`` on
    ties."""
    if not candidates:
        return -1
    return min(candidates, key=lambda j: (abs(j - expected), -(j >= expected)))


# ── core formatter ───────────────────────────────────────────────────


def _try_align_box(lines, start, end, left_col, all_boxes):
    """Align a single box.

    Returns ``True`` if any line was modified.
    """

    edges = {}  # line_idx → ("border"|"middle", col)

    # top border — first ┐ after ┌
    r_top = _first_char(lines[start], left_col, "┐")
    # bottom border — first ┘ after └
    r_bot = _first_char(lines[end], left_col, "┘")
    if r_top < 0 or r_bot < 0:
        return False

    edges[start] = ("border", r_top)
    edges[end] = ("border", r_bot)

    # Use display columns for candidate selection (CJK-safe)
    expected_dc = max(
        _display_width(lines[start][left_col:r_top]),
        _display_width(lines[end][left_col:r_bot]),
    )

    def _pick(line, cands):
        """Pick the candidate whose display column is closest to.

        *expected_dc*.
        """
        return min(
            cands,
            key=lambda j: (
                abs(_display_width(line[left_col:j]) - expected_dc),
                -(_display_width(line[left_col:j]) >= expected_dc),
            ),
        )

    for i in range(start + 1, end):
        if left_col >= len(lines[i]):
            continue
        ch = lines[i][left_col]
        if ch in ("│", "┤", "┼"):
            cands = [j for j in range(left_col + 1, len(lines[i])) if lines[i][j] in "│┤"]
            if cands:
                edges[i] = ("middle", _pick(lines[i], cands))
        elif ch == "├":
            cands_div = [j for j in range(left_col + 1, len(lines[i])) if lines[i][j] == "┤"]
            if cands_div:
                edges[i] = ("border", _pick(lines[i], cands_div))
            else:
                # tree-style list item (├─ text │) — treat as middle
                cands_mid = [j for j in range(left_col + 1, len(lines[i])) if lines[i][j] in "│┤"]
                if cands_mid:
                    edges[i] = ("middle", _pick(lines[i], cands_mid))

    if len(edges) < 2:
        return False

    # ── compute target display width ──
    # Display width of each line span from left_col to right edge (inclusive).
    line_dws = {}
    for i, (tp, col) in edges.items():
        line_dws[i] = _display_width(lines[i][left_col : col + 1])

    max_border_dw = 0
    max_content_dw = 2  # minimum: │ + │
    for i, (tp, col) in edges.items():
        if tp == "border":
            max_border_dw = max(max_border_dw, line_dws[i])
        else:
            content = lines[i][left_col + 1 : col]
            stripped, stripped_dw = _rstrip_display(content)
            # +2 for left and right edge characters (each 1 display column)
            min_dw = stripped_dw + 2
            max_content_dw = max(max_content_dw, min_dw)

    target_dw = max(max_border_dw, max_content_dw)

    # ── apply adjustments ──
    changed = False
    for i, (tp, col) in edges.items():
        delta = target_dw - line_dws[i]  # in display columns
        if delta == 0:
            continue
        changed = True
        line = lines[i]
        rc = line[col]  # the right-edge character itself
        after = line[col + 1 :]

        if tp == "border":
            before = line[:col]
            if delta > 0:
                lines[i] = before + "─" * delta + rc + after
            else:
                # shrink — remove trailing ─ only
                b = list(before)
                trim = -delta
                while trim > 0 and b and b[-1] == "─":
                    b.pop()
                    trim -= 1
                lines[i] = "".join(b) + rc + after
        else:
            # middle line — rebuild content to exact target display width
            content = lines[i][left_col + 1 : col]
            stripped, stripped_dw = _rstrip_display(content)
            # target content display width = target_dw - left_edge(1) - right_edge(1)
            target_content_dw = target_dw - 2
            padding = target_content_dw - stripped_dw
            new_content = stripped + " " * max(0, padding)
            lines[i] = line[: left_col + 1] + new_content + rc + after

    return changed


def _format_block(lines):
    """Align right edges of every box in *lines* (mutated in place)."""
    for _ in range(80):  # generous convergence budget
        boxes = _find_boxes(lines)
        if not boxes:
            break

        # process the deepest (innermost) box first so that outer boxes
        # pick up any width changes on the next iteration
        boxes.sort(key=lambda b: -_nesting_depth(b, boxes))

        made_progress = False
        for start, end, left_col in boxes:
            if _try_align_box(lines, start, end, left_col, boxes):
                made_progress = True
                break  # restart – positions may have shifted

        if not made_progress:
            break

    return lines


# ── file-level driver ────────────────────────────────────────────────


def _process_file(path, *, check=False):
    with open(path, encoding="utf-8") as fh:
        original = fh.read()

    lines = original.split("\n")
    blocks = _find_code_blocks(lines)
    if not blocks:
        return False

    for bstart, bend in blocks:
        block = lines[bstart:bend]
        _format_block(block)
        lines[bstart:bend] = block

    result = "\n".join(lines)
    if result == original:
        return False

    if not check:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(result)
    return True


def main():
    ap = argparse.ArgumentParser(description="Format ASCII box-drawing diagrams in Markdown.")
    ap.add_argument("files", nargs="*", help="Markdown files to format")
    ap.add_argument("--check", action="store_true", help="Report files that need formatting without modifying them")
    args = ap.parse_args()

    ret = 0
    for path in args.files:
        if not path.endswith(".md"):
            continue
        if _process_file(path, check=args.check):
            action = "needs formatting" if args.check else "formatted"
            print(f"{action}: {path}")
            ret = 1

    raise SystemExit(ret)


if __name__ == "__main__":
    main()
