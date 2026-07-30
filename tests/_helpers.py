# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared helpers for the test suite.

A plain module rather than conftest.py, because conftest is loaded by pytest as
a plugin and `tests/` is not a package — importing from it directly is not
portable. pytest prepends this directory to sys.path, so a flat import works.
"""

from __future__ import annotations

import re

# Rich styles option flags in pieces: `--json` renders as
# "\x1b[1;2;36m-\x1b[0m\x1b[1;2;36m-json\x1b[0m", so the literal substring is
# absent from styled output. Colour is off when stdout is not a terminal and on
# under CI, which is why assertions on help text passed locally and failed on
# every runner. Strip escapes before asserting on any CLI output.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def plain(text: str) -> str:
    """CLI output with terminal escape sequences removed."""
    return _ANSI.sub("", text)
