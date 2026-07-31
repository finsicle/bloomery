#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Every outstanding review finding on a pull request, including the ones the
# merge rule cannot see.
#
# GitHub can only anchor a review comment to a line inside a diff hunk, so a
# reviewer that reads whole files has nowhere to put a finding about code the
# current diff does not touch. Those land in the review *body* under "Outside
# diff range comments": no comment id, not a review thread, and therefore
# invisible to the "all conversations resolved" branch rule.
#
# One such finding — a Major bug that let a cancel kill an unrelated process —
# sat in a PR that reported zero unresolved threads. This script exists so that
# cannot happen quietly again.
#
# Usage: scripts/review-findings.sh [pr-number]

set -euo pipefail

missing=()
for tool in gh jq python3; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "error: not on PATH: ${missing[*]}" >&2
    echo "  gh      https://cli.github.com" >&2
    echo "  jq      https://jqlang.github.io/jq/" >&2
    echo "  python3 https://www.python.org/downloads/" >&2
    exit 1
fi

PR="${1:-$(gh pr view --json number -q .number)}"
REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"

echo "pull request #${PR}  (${REPO})"
echo

# Paginated deliberately. A tool whose whole purpose is "no finding goes
# unnoticed" must not stop at the first hundred.
unresolved="$(
  gh api graphql --paginate \
    -f query='
    query($owner: String!, $name: String!, $pr: Int!, $endCursor: String) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $pr) {
          reviewThreads(first: 100, after: $endCursor) {
            pageInfo { hasNextPage endCursor }
            nodes { isResolved path line comments(first: 1) { nodes { databaseId } } }
          }
        }
      }
    }' \
    -F owner="${REPO%%/*}" -F name="${REPO##*/}" -F pr="${PR}" \
    --jq '.data.repository.pullRequest.reviewThreads.nodes[]
          | select(.isResolved | not)
          | "  \(.comments.nodes[0].databaseId)  \(.path):\(.line // "?")"'
)"

if [ -z "$unresolved" ]; then
    echo "inline threads, unresolved: 0"
else
    echo "inline threads, unresolved: $(printf '%s\n' "$unresolved" | wc -l | tr -d ' ')"
    printf '%s\n' "$unresolved"
fi

echo
echo "outside-diff findings (invisible to the merge rule):"
gh api "repos/${REPO}/pulls/${PR}/reviews" --paginate --jq '.[].body' |
  python3 -c '
import re
import sys

text = sys.stdin.read()
blocks = re.findall(r"Outside diff range comments.*?(?=</details>\s*</details>|\Z)", text, re.DOTALL)
seen = set()
found = False

for block in blocks:
    # Each finding starts with a `line-range`: prefix inside the collapsed block.
    for line, body in re.findall(r"`([\d-]+)`: (.*?)(?=\n> `[\d-]+`: |\Z)", block, re.DOTALL):
        if line in seen:
            continue
        seen.add(line)
        found = True
        title = re.search(r"\*\*(.+?)\*\*", body)
        print(f"  {line}: {title.group(1) if title else body.strip()[:100]}")

if not found:
    print("  none")
'

echo
echo "These are reported for review, not merge-blocking. Read them before merging."
