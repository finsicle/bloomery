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

PR="${1:-$(gh pr view --json number -q .number)}"
REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"

echo "pull request #${PR}  (${REPO})"
echo

unresolved="$(
  gh api graphql -f query='
    query($owner: String!, $name: String!, $pr: Int!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $pr) {
          reviewThreads(first: 100) {
            nodes { isResolved path line comments(first: 1) { nodes { databaseId } } }
          }
        }
      }
    }' \
    -F owner="${REPO%%/*}" -F name="${REPO##*/}" -F pr="${PR}" \
    --jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved | not)]'
)"

count="$(printf '%s' "$unresolved" | jq 'length')"
echo "inline threads, unresolved: ${count}"
printf '%s' "$unresolved" |
  jq -r '.[] | "  \(.comments.nodes[0].databaseId)  \(.path):\(.line // "?")"'

echo
echo "outside-diff findings (invisible to the merge rule):"
gh api "repos/${REPO}/pulls/${PR}/reviews" --paginate --jq '.[].body' |
  python3 -c '
import re
import sys

text = sys.stdin.read()
blocks = re.findall(r"Outside diff range comments.*?(?=</details>\s*</details>|\Z)", text, re.DOTALL)
if not blocks:
    print("  none")
    sys.exit()

for block in blocks:
    # Each finding starts with a `line-range`: prefix inside the collapsed block.
    for line, body in re.findall(r"`([\d-]+)`: (.*?)(?=\n> `[\d-]+`: |\Z)", block, re.DOTALL):
        title = re.search(r"\*\*(.+?)\*\*", body)
        print(f"  {line}: {title.group(1) if title else body.strip()[:100]}")
'

echo
echo "These are reported for review, not merge-blocking. Read them before merging."
