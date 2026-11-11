#!/usr/bin/env python3
"""approval-guard: approvals from people who touched a PR do not count.

A PR into the default branch passes only if at least one human with write
access approved its current head commit and did not touch the PR. "Touched"
means any of:
  - the PR author;
  - the author or committer of any commit in the PR;
  - anyone who pushed to the PR's head branch, taken from an unbroken chain
    of repository-activity entries from branch creation up to the head;
  - whoever created any commit that GitHub signed on someone's behalf
    (committer "web-flow"); their author field is caller-chosen, so the
    creator comes from the activity entry that introduced that commit.

Fails closed: an incomplete push record, an unsigned commit, a bot commit
or push (other than Dependabot), a fork PR, too many commits, and any
error all give a failing check (or no check, which also blocks the merge
once the check is required). Standard library only. Never runs PR code.
"""
from __future__ import annotations

import datetime as dt
import json
import time
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Iterable

CHECK_NAME = "approval-guard"
ALLOWED_BOTS = {"dependabot[bot]"}
WEB_FLOW = "web-flow"
WRITE_PERMISSIONS = {"admin", "write"}  # "maintain" reports as "write"
MAX_COMMITS = 250  # the PR commits API returns at most 250
DECIDING_STATES = {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}
ZERO_SHA = "0" * 40
WEB_FLOW_LOOKBACK = dt.timedelta(days=1)
MAX_CLOCK_SKEW = dt.timedelta(minutes=5)
MAX_REPO_ACTIVITY_PAGES = 30  # 3000 entries; beyond that we fail closed
HEAD_RETRIES = 3
HEAD_RETRY_SECONDS = 10


class GuardError(Exception):
    """Anything that must turn into a failing check."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class StaleRecord(GuardError):
    """The activity log has not caught up with the current head yet."""


# --------------------------------------------------------------------------
# Decision logic (pure, unit-tested)
# --------------------------------------------------------------------------

def _login(user: dict | None) -> str | None:
    if not user or not user.get("login"):
        return None
    return user["login"].lower()


def _is_disallowed_bot(user: dict) -> bool:
    return user.get("type") == "Bot" and user["login"].lower() not in ALLOWED_BOTS


def push_chain(entries: list[dict], head: str) -> list[dict]:
    """Return the activity entries that built the branch, newest first.

    `entries` are the activity entries for the PR's head ref, newest first.
    The chain must start at the current head, link each entry's `before`
    to the next older entry's `after`, and end at a branch_creation. Older
    entries (an earlier branch with the same name) are dropped. Anything
    else raises: the record is stale, has gaps, or the branch was renamed.
    """
    seen: set = set()
    deduped = []
    for e in entries:  # tolerate the same entry listed twice
        if e.get("id") is not None and e["id"] in seen:
            continue
        seen.add(e.get("id"))
        deduped.append(e)
    entries = deduped
    if not entries:
        raise StaleRecord("no push record for the PR branch")
    if entries[0].get("after") != head:
        raise StaleRecord("push record does not reach the current head yet; re-run shortly")
    chain = []
    for i, e in enumerate(entries):
        chain.append(e)
        if e.get("activity_type") == "branch_creation":
            if e.get("before") != ZERO_SHA:
                raise GuardError("malformed branch_creation entry")
            return chain
        nxt = entries[i + 1] if i + 1 < len(entries) else None
        if nxt is None or nxt.get("after") != e.get("before"):
            raise GuardError("push record for the PR branch has a gap (renamed branch?)")
    raise GuardError("push record has no branch creation")


def web_flow_creators(commits: list[dict], chain: list[dict],
                      repo_entries: list[dict]) -> tuple[list[dict], list[str]]:
    """Find who created each GitHub-signed (committer web-flow) commit.

    Returns (extra activity entries to treat as pushes, problems). A
    web-flow commit that was never the tip of a push anywhere in the repo
    cannot be attributed, so it is a problem.
    """
    by_after: dict[str, list[dict]] = {}
    for e in list(chain) + list(repo_entries):
        by_after.setdefault(e.get("after"), []).append(e)
    extra, problems = [], []
    for c in commits:
        if _login(c.get("committer")) != WEB_FLOW:
            continue
        # Credit EVERY push that ever pointed a branch at this commit, so
        # the push that created it is counted even if someone else later
        # fast-forwarded another branch to it.
        found = by_after.get(c["sha"])
        if not found:
            problems.append(f"cannot tell who created web/API commit {c['sha'][:8]}")
        else:
            extra.extend(found)
    return extra, problems


def touched_by(pr: dict, commits: list[dict], pushes: list[dict]) -> tuple[set[str], list[str]]:
    """Return (logins who touched the PR, problems that force a failure)."""
    touched: set[str] = set()
    problems: list[str] = []

    author = _login(pr.get("user"))
    if author is None:
        problems.append("PR author is unknown")
    else:
        touched.add(author)

    for c in commits:
        sha = c.get("sha", "?")[:8]
        if not c.get("commit", {}).get("verification", {}).get("verified"):
            problems.append(f"commit {sha} has no verified signature")
        # The committer is who signed the commit, so it must be a known human
        # (or Dependabot), or GitHub itself (web-flow; creator found via pushes).
        committer = c.get("committer")
        login = _login(committer)
        if login is None:
            problems.append(f"commit {sha}: committer is not linked to a GitHub account")
        elif login != WEB_FLOW:
            if _is_disallowed_bot(committer):
                problems.append(f"commit {sha}: committer is bot {committer['login']}")
            else:
                touched.add(login)
        # The author is chosen by whoever made the commit: count a linked
        # human author as touched (it only removes approvers), ignore others.
        a = c.get("author")
        if _login(a) and a.get("type") == "User":
            touched.add(_login(a))

    for p in pushes:
        actor = p.get("actor")
        login = _login(actor)
        if login is None:
            problems.append("a push to the PR has no known actor")
        elif _is_disallowed_bot(actor):
            problems.append(f"push by bot {actor['login']}")
        else:
            touched.add(login)

    return touched, problems


def latest_reviews(reviews: Iterable[dict]) -> dict[str, dict]:
    """Latest deciding review per reviewer (COMMENTED/PENDING ignored)."""
    latest: dict[str, dict] = {}
    for r in sorted(reviews, key=lambda r: r["id"]):
        login = _login(r.get("user"))
        if login is None or r.get("state") not in DECIDING_STATES:
            continue
        latest[login] = r
    return latest


def decide(
    pr: dict,
    commits: list[dict],
    reviews: list[dict],
    pushes: list[dict],
    permission_of: Callable[[str], str],
) -> tuple[str, str, str]:
    """Return (conclusion, title, summary) for the check run."""
    head = pr["head"]["sha"]
    touched, problems = touched_by(pr, commits, pushes)
    if problems:
        return ("failure", "Cannot verify who touched this PR",
                "\n".join(f"- {p}" for p in problems))

    def can_write(login: str) -> bool | None:
        try:
            return permission_of(login) in WRITE_PERMISSIONS
        except GuardError:
            return None

    latest = latest_reviews(reviews)
    blockers = []
    for login, r in sorted(latest.items()):
        if r["state"] != "CHANGES_REQUESTED" or r["user"].get("type") != "User":
            continue
        if can_write(login) is not False:  # writer, or unknown: fail closed
            blockers.append(login)
    if blockers:
        return ("failure", "Changes requested", "Changes requested by: " + ", ".join(blockers))

    eligible, ignored = [], []
    for login, r in sorted(latest.items()):
        if r["state"] != "APPROVED" or r.get("commit_id") != head:
            continue
        if r["user"].get("type") != "User":
            ignored.append(f"{login} (bot)")
        elif login in touched:
            ignored.append(f"{login} (touched the PR)")
        else:
            w = can_write(login)
            if w is None:
                ignored.append(f"{login} (permission lookup failed)")
            elif not w:
                ignored.append(f"{login} (no write access)")
            else:
                eligible.append(login)

    summary = "Touched this PR (approvals do not count): " + ", ".join(sorted(touched))
    if ignored:
        summary += "\nIgnored approvals on this commit: " + ", ".join(ignored)
    if eligible:
        return ("success", "Approved by " + ", ".join(eligible), summary)
    return ("failure", "Needs an approval from someone who did not touch this PR", summary)


# --------------------------------------------------------------------------
# GitHub API
# --------------------------------------------------------------------------

API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')


def _request(url: str, method: str = "GET", body: dict | None = None) -> tuple[object, str]:
    token = os.environ["GH_TOKEN"]
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "approval-guard",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read() or b"null"), resp.headers.get("Link", "") or ""
    except urllib.error.HTTPError as e:
        raise GuardError(f"{method} {url} -> HTTP {e.code}", e.code) from e
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise GuardError(f"{method} {url} -> {e}") from e


def get(path: str):
    return _request(f"{API}{path}")[0]


def get_pages(path: str, stop: Callable[[dict], bool] | None = None,
              max_pages: int | None = None, require_stop: bool = False) -> list:
    """GET every page (Link rel=next). Stops after a page where stop(item).

    With max_pages, reaching the cap without stopping raises (fail closed).
    With require_stop, running out of pages before stop fires raises: the
    history we needed may have expired.
    """
    sep = "&" if "?" in path else "?"
    url, out, pages = f"{API}{path}{sep}per_page=100", [], 0
    while url:
        if max_pages is not None and pages >= max_pages:
            raise GuardError(f"too much activity to verify ({path})")
        pages += 1
        page, link = _request(url)
        if not isinstance(page, list):
            raise GuardError(f"expected a list from {path}")
        out.extend(page)
        if stop and any(stop(item) for item in page):
            return out
        m = _NEXT.search(link)
        url = m.group(1) if m else None
    if require_stop:
        raise GuardError(f"history ended before the needed window ({path})")
    return out


def post_check(repo: str, sha: str, conclusion: str, title: str, summary: str) -> None:
    _request(f"{API}/repos/{repo}/check-runs", "POST", {
        "name": CHECK_NAME,
        "head_sha": sha,
        "status": "completed",
        "conclusion": conclusion,
        "output": {"title": title[:200], "summary": summary[:60000]},
    })


def _ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def candidate_prs(repo: str, head_sha: str, pr_hint: str, run_prs: str) -> set[int]:
    nums: set[int] = set()
    if pr_hint:
        nums.add(int(pr_hint))
    for p in json.loads(run_prs or "[]") or []:
        if isinstance(p, dict) and isinstance(p.get("number"), int):
            nums.add(p["number"])
    if not nums and head_sha:
        nums.update(p["number"] for p in get(f"/repos/{repo}/commits/{head_sha}/pulls"))
    return nums


def gather(repo: str, number: int, pr: dict) -> tuple[list, list, list]:
    """Fetch commits, reviews and the verified push record for a PR."""
    head = pr["head"]["sha"]
    if (pr["head"].get("repo") or {}).get("full_name") != repo:
        raise GuardError("PRs from forks are not supported; push the branch to this repo")
    if pr["commits"] > MAX_COMMITS:
        raise GuardError(f"PR has more than {MAX_COMMITS} commits")
    commits = get_pages(f"/repos/{repo}/pulls/{number}/commits")
    if len(commits) != pr["commits"]:
        raise GuardError("commit list is incomplete")
    reviews = get_pages(f"/repos/{repo}/pulls/{number}/reviews")

    ref = urllib.parse.quote(f"refs/heads/{pr['head']['ref']}", safe="")
    for attempt in range(HEAD_RETRIES):
        entries = get_pages(f"/repos/{repo}/activity?ref={ref}&direction=desc",
                            stop=lambda e: e.get("activity_type") == "branch_creation")
        try:
            chain = push_chain(entries, head)
            break
        except StaleRecord:
            if attempt == HEAD_RETRIES - 1:
                raise
            time.sleep(HEAD_RETRY_SECONDS)

    pushes = list(chain)
    web = [c for c in commits if _login(c.get("committer")) == WEB_FLOW]
    if web:
        now = dt.datetime.now(dt.timezone.utc)
        dates = []
        for c in web:
            signed = c["commit"].get("verification", {}).get("verified_at")
            if not signed:
                raise GuardError(f"web/API commit {c['sha'][:8]} has no signing time")
            committed = _ts(c["commit"]["committer"]["date"])
            if committed > now + MAX_CLOCK_SKEW:
                raise GuardError("a web/API commit is dated in the future")
            # verified_at is set by GitHub when it signs; the committer date
            # may be caller-influenced, so anchor on the earlier of the two.
            dates.append(min(committed, _ts(signed)))
        since = min(dates) - WEB_FLOW_LOOKBACK
        repo_entries = get_pages(f"/repos/{repo}/activity?direction=desc",
                                 stop=lambda e: _ts(e["timestamp"]) < since,
                                 max_pages=MAX_REPO_ACTIVITY_PAGES, require_stop=True)
        extra, problems = web_flow_creators(commits, chain, repo_entries)
        if problems:
            raise GuardError("; ".join(problems))
        pushes += extra
    return commits, reviews, pushes


def evaluate_pr(repo: str, default_branch: str, number: int) -> None:
    pr = get(f"/repos/{repo}/pulls/{number}")
    if pr["state"] != "open" or pr["base"]["ref"] != default_branch:
        print(f"PR #{number}: not an open PR into {default_branch}; skipped")
        return
    head = pr["head"]["sha"]
    try:
        commits, reviews, pushes = gather(repo, number, pr)
        perms: dict[str, str] = {}

        def permission_of(login: str) -> str:
            if login not in perms:
                try:
                    perms[login] = get(f"/repos/{repo}/collaborators/{login}/permission")["permission"]
                except GuardError as e:
                    if e.status != 404:  # 404: not a collaborator / account gone
                        raise
                    perms[login] = "none"
            return perms[login]

        conclusion, title, summary = decide(pr, commits, reviews, pushes, permission_of)
    except Exception as e:  # noqa: BLE001 - every failure must become a failing check
        conclusion, title, summary = "failure", "approval-guard could not evaluate this PR", str(e)

    # Post only if the head did not move while we were evaluating; the run
    # triggered by the newer push will post for the new head.
    if get(f"/repos/{repo}/pulls/{number}")["head"]["sha"] != head:
        print(f"PR #{number}: head moved during evaluation; skipped")
        return
    post_check(repo, head, conclusion, title, summary)
    print(f"PR #{number} @ {head[:8]}: {conclusion}: {title}")


def main() -> int:
    repo = os.environ["REPO"]
    default_branch = get(f"/repos/{repo}")["default_branch"]
    numbers = candidate_prs(repo, os.environ.get("HEAD_SHA", ""),
                            os.environ.get("PR_HINT", ""), os.environ.get("RUN_PRS", ""))
    if not numbers:
        print("no PR found for this run; nothing to do")
        return 0
    failed = False
    for n in sorted(numbers):
        try:
            evaluate_pr(repo, default_branch, n)
        except Exception as e:  # noqa: BLE001
            # Could not post a check: the required check stays missing (or
            # stale on an old head), so the merge stays blocked. Go red.
            print(f"PR #{n}: {e}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
