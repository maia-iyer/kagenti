#!/usr/bin/env python3
"""Analyze Claude Code JSONL session files and produce structured analytics.

Pipeline phases:
  stats     - Parse JSONL, extract token usage, tools, costs -> JSON
  mermaid   - Generate workflow diagram from stats (placeholder)
  comment   - Post PR/issue comment with session summary (placeholder)
  summary   - Generate human-readable summary (placeholder)
  extract   - Export sessions to CSV/MD/HTML (placeholder)
  dashboard - Aggregate multi-session dashboard (placeholder)

Usage:
    python3 .claude/scripts/session-analytics.py \\
        --phase stats --session-id <UUID>

    python3 .claude/scripts/session-analytics.py \\
        --phase stats --session-id <UUID> --stats-file /tmp/out.json

    python3 .claude/scripts/session-analytics.py --self-test
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"

OUTPUT_BASE = Path("/tmp/kagenti/session")

# Pricing per 1M tokens (USD)
MODEL_PRICING = {
    "claude-opus-4-6": {
        "input": 15.0,
        "output": 75.0,
        "cache_create": 18.75,
        "cache_read": 1.50,
    },
    "claude-opus-4-5-20251101": {
        "input": 15.0,
        "output": 75.0,
        "cache_create": 18.75,
        "cache_read": 1.50,
    },
    "claude-sonnet-4-5-20250514": {
        "input": 3.0,
        "output": 15.0,
        "cache_create": 3.75,
        "cache_read": 0.30,
    },
    "claude-haiku-4-5-20251001": {
        "input": 0.80,
        "output": 4.0,
        "cache_create": 1.0,
        "cache_read": 0.08,
    },
}

# Fallback pricing for unknown models (use opus pricing as conservative default)
FALLBACK_PRICING = {
    "input": 15.0,
    "output": 75.0,
    "cache_create": 18.75,
    "cache_read": 1.50,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Analyze Claude Code JSONL session files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--phase",
        choices=["stats", "mermaid", "comment", "summary", "extract", "dashboard"],
        required=False,
        help="Pipeline phase to execute",
    )
    p.add_argument("--session-id", help="Session UUID to analyze")
    p.add_argument(
        "--project-dir",
        help="Claude Code project dir (default: auto-detect from ~/.claude/projects/)",
    )
    p.add_argument(
        "--stats-file",
        help="Path to stats JSON (input for downstream phases, output for stats phase)",
    )
    p.add_argument("--skills-dir", help="Path to .claude/skills/ directory")
    p.add_argument(
        "--target",
        choices=["pr", "issue"],
        help="GitHub target type for comment phase",
    )
    p.add_argument("--number", type=int, help="PR or issue number")
    p.add_argument("--repo", help="GitHub repo (OWNER/NAME)")
    p.add_argument("--from-date", help="Start date for extract (YYYY-MM-DD)")
    p.add_argument("--to-date", help="End date for extract (YYYY-MM-DD)")
    p.add_argument(
        "--output-dir",
        default=str(OUTPUT_BASE),
        help=f"Output directory (default: {OUTPUT_BASE})",
    )
    p.add_argument(
        "--self-test", action="store_true", help="Run built-in self-test mode"
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# JSONL discovery
# ---------------------------------------------------------------------------


def find_session_jsonl(session_id, project_dir=None):
    """Locate the JSONL file for a given session UUID.

    Search order:
    1. If project_dir is given, look there directly.
    2. Scan all directories under ~/.claude/projects/ for <session_id>.jsonl
    """
    filename = f"{session_id}.jsonl"

    if project_dir:
        candidate = Path(project_dir) / filename
        if candidate.is_file():
            return candidate
        # Maybe project_dir is already the full projects subdir
        for p in Path(project_dir).rglob(filename):
            return p
        return None

    # Auto-detect: scan all project dirs
    if not CLAUDE_PROJECTS_DIR.is_dir():
        return None

    for proj in CLAUDE_PROJECTS_DIR.iterdir():
        if proj.is_dir():
            candidate = proj / filename
            if candidate.is_file():
                return candidate

    return None


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------


def parse_timestamp(ts_str):
    """Parse an ISO timestamp string into a datetime object."""
    if not ts_str:
        return None
    try:
        # Handle both 'Z' suffix and '+00:00'
        ts_str = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


def extract_commit_message(command_str):
    """Extract the commit message from a git commit command string."""
    if not command_str or "git commit" not in command_str:
        return None

    # Try to extract from -m flag with heredoc pattern
    # Pattern: git commit -m "$(cat <<'EOF'\n<message>\nEOF\n)"
    heredoc_match = re.search(
        r"git commit.*?-m\s+\"\$\(cat <<'EOF'\n(.*?)\nEOF", command_str, re.DOTALL
    )
    if heredoc_match:
        msg = heredoc_match.group(1).strip()
        # Take just the first line as the message
        return msg.split("\n")[0].strip()

    # Pattern: git commit -m "message"
    quoted_match = re.search(r'git commit.*?-m\s+"([^"]+)"', command_str)
    if quoted_match:
        return quoted_match.group(1).split("\n")[0].strip()

    # Pattern: git commit -m 'message'
    single_match = re.search(r"git commit.*?-m\s+'([^']+)'", command_str)
    if single_match:
        return single_match.group(1).split("\n")[0].strip()

    return None


def parse_session_jsonl(jsonl_path):
    """Parse a Claude Code JSONL session file and return structured stats.

    Returns a dict matching the output JSON schema.
    """
    models = defaultdict(
        lambda: {
            "messages": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
        }
    )
    tool_counts = Counter()
    skill_invocations = Counter()
    subagents = {}  # agentId -> info dict
    commits = []
    timestamps = []

    session_id = None
    git_branch = None

    with open(jsonl_path, "r") as f:
        for line_num, raw_line in enumerate(f, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                print(
                    f"WARNING: skipping malformed JSON at line {line_num}",
                    file=sys.stderr,
                )
                continue

            rec_type = record.get("type")

            # Extract session metadata from first user/assistant records
            if session_id is None and record.get("sessionId"):
                session_id = record["sessionId"]
            if git_branch is None and record.get("gitBranch"):
                git_branch = record["gitBranch"]

            # Collect timestamps from all records that have them
            ts = parse_timestamp(record.get("timestamp"))
            if ts:
                timestamps.append(ts)

            # --- Assistant records: model, usage, tool_use ---
            if rec_type == "assistant":
                message = record.get("message", {})
                model = message.get("model")
                usage = message.get("usage", {})
                agent_id = record.get("agentId")

                if model:
                    m = models[model]
                    m["messages"] += 1
                    m["input_tokens"] += usage.get("input_tokens", 0)
                    m["output_tokens"] += usage.get("output_tokens", 0)
                    m["cache_creation_tokens"] += usage.get(
                        "cache_creation_input_tokens", 0
                    )
                    m["cache_read_tokens"] += usage.get("cache_read_input_tokens", 0)

                # Track subagent tokens
                if agent_id and model:
                    if agent_id not in subagents:
                        subagents[agent_id] = {
                            "id": agent_id[:7] if len(agent_id) > 7 else agent_id,
                            "type": "unknown",
                            "model": model,
                            "tokens": {"input": 0, "output": 0},
                            "description": "",
                        }
                    sa = subagents[agent_id]
                    sa["tokens"]["input"] += usage.get("input_tokens", 0)
                    sa["tokens"]["output"] += usage.get("output_tokens", 0)

                # Parse tool_use content blocks
                content = message.get("content", [])
                if not isinstance(content, list):
                    continue

                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_use":
                        continue

                    tool_name = block.get("name", "unknown")
                    tool_counts[tool_name] += 1
                    tool_input = block.get("input", {})

                    # Skill invocations
                    if tool_name == "Skill":
                        skill_name = tool_input.get("skill", "unknown")
                        skill_invocations[skill_name] += 1

                    # Subagent launches via Task tool
                    if tool_name == "Task":
                        sub_type = tool_input.get("subagent_type", "unknown")
                        desc = tool_input.get("description", "")
                        sub_model = tool_input.get("model", "")
                        # Use a hash of the description as a pseudo-id
                        pseudo_id = f"task-{len(subagents)}"
                        subagents[pseudo_id] = {
                            "id": pseudo_id,
                            "type": sub_type,
                            "model": sub_model,
                            "tokens": {"input": 0, "output": 0},
                            "description": desc[:120],
                        }

                    # Git commits via Bash
                    if tool_name == "Bash":
                        cmd = tool_input.get("command", "")
                        commit_msg = extract_commit_message(cmd)
                        if commit_msg:
                            commits.append({"message": commit_msg})

    # --- Compute aggregates ---
    total_input = sum(m["input_tokens"] for m in models.values())
    total_output = sum(m["output_tokens"] for m in models.values())
    total_cache_creation = sum(m["cache_creation_tokens"] for m in models.values())
    total_cache_read = sum(m["cache_read_tokens"] for m in models.values())

    # Timestamps
    started_at = min(timestamps) if timestamps else None
    ended_at = max(timestamps) if timestamps else None
    duration_minutes = 0
    if started_at and ended_at:
        duration_minutes = round((ended_at - started_at).total_seconds() / 60)

    # Cost estimation
    estimated_cost = 0.0
    for model_name, m in models.items():
        pricing = MODEL_PRICING.get(model_name, FALLBACK_PRICING)
        cost = (
            m["input_tokens"] * pricing["input"] / 1_000_000
            + m["output_tokens"] * pricing["output"] / 1_000_000
            + m["cache_creation_tokens"] * pricing["cache_create"] / 1_000_000
            + m["cache_read_tokens"] * pricing["cache_read"] / 1_000_000
        )
        estimated_cost += cost

    # Build skills list
    skills_list = [
        {"skill": name, "count": count, "status": "unknown"}
        for name, count in skill_invocations.most_common()
    ]

    # Build subagents list
    subagents_list = list(subagents.values())

    # Deduplicate commits by message
    seen_commits = set()
    unique_commits = []
    for c in commits:
        if c["message"] not in seen_commits:
            seen_commits.add(c["message"])
            unique_commits.append(c)

    # Use provided session_id or derive from filename
    if not session_id:
        session_id = jsonl_path.stem

    session_id_short = session_id[:8] if session_id else "unknown"

    return {
        "session_id": session_id,
        "session_id_short": session_id_short,
        "branch": git_branch or "",
        "started_at": started_at.isoformat() if started_at else None,
        "ended_at": ended_at.isoformat() if ended_at else None,
        "duration_minutes": duration_minutes,
        "models": {name: dict(data) for name, data in models.items()},
        "total_tokens": {
            "input": total_input,
            "output": total_output,
            "cache_creation": total_cache_creation,
            "cache_read": total_cache_read,
        },
        "estimated_cost_usd": round(estimated_cost, 2),
        "tools": dict(tool_counts.most_common()),
        "skills_invoked": skills_list,
        "subagents": subagents_list,
        "commits": unique_commits,
        "problems_faced": [],
        "workflow_edges": {},
    }


# ---------------------------------------------------------------------------
# Phase: stats
# ---------------------------------------------------------------------------


def phase_stats(args):
    """Parse session JSONL and output structured stats JSON."""
    if not args.session_id:
        print("ERROR: --session-id is required for stats phase", file=sys.stderr)
        sys.exit(1)

    jsonl_path = find_session_jsonl(args.session_id, args.project_dir)
    if not jsonl_path:
        print(
            f"ERROR: session JSONL not found for {args.session_id}",
            file=sys.stderr,
        )
        print(
            f"Searched in: {args.project_dir or CLAUDE_PROJECTS_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Parsing: {jsonl_path}", file=sys.stderr)

    stats = parse_session_jsonl(jsonl_path)

    # Determine output path
    if args.stats_file:
        output_path = Path(args.stats_file)
    else:
        output_dir = Path(args.output_dir)
        output_path = output_dir / f"{stats['session_id_short']}-stats.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Stats written to: {output_path}", file=sys.stderr)

    # Print summary to stderr
    print(f"\n--- Session Summary ---", file=sys.stderr)
    print(f"Session:    {stats['session_id_short']}", file=sys.stderr)
    print(f"Branch:     {stats['branch']}", file=sys.stderr)
    print(f"Duration:   {stats['duration_minutes']} minutes", file=sys.stderr)
    print(f"Models:     {', '.join(stats['models'].keys())}", file=sys.stderr)
    total = stats["total_tokens"]
    print(
        f"Tokens:     {total['input']:,} in / {total['output']:,} out / "
        f"{total['cache_creation']:,} cache-create / {total['cache_read']:,} cache-read",
        file=sys.stderr,
    )
    print(f"Cost:       ${stats['estimated_cost_usd']:.2f}", file=sys.stderr)
    print(
        f"Tools:      {sum(stats['tools'].values())} calls across {len(stats['tools'])} tools",
        file=sys.stderr,
    )
    if stats["skills_invoked"]:
        print(
            f"Skills:     {', '.join(s['skill'] for s in stats['skills_invoked'])}",
            file=sys.stderr,
        )
    if stats["subagents"]:
        print(f"Subagents:  {len(stats['subagents'])} launched", file=sys.stderr)
    if stats["commits"]:
        print(f"Commits:    {len(stats['commits'])}", file=sys.stderr)

    # Also print JSON to stdout for piping
    print(json.dumps(stats, indent=2))

    return stats


# ---------------------------------------------------------------------------
# Phase: placeholders for future phases
# ---------------------------------------------------------------------------


def phase_placeholder(phase_name, args):
    """Placeholder for phases not yet implemented."""
    print(f"Phase '{phase_name}' is not yet implemented.", file=sys.stderr)
    print("Available phases: stats", file=sys.stderr)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def run_self_test():
    """Run basic self-tests to verify the script works correctly."""
    import tempfile

    print("Running self-tests...", file=sys.stderr)
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name}", file=sys.stderr)
            passed += 1
        else:
            print(f"  FAIL: {name} {detail}", file=sys.stderr)
            failed += 1

    # Test 1: parse_timestamp
    ts = parse_timestamp("2026-01-19T10:06:49.390Z")
    check("parse_timestamp with Z", ts is not None and ts.year == 2026)

    ts2 = parse_timestamp("2026-01-19T10:06:49.390+00:00")
    check("parse_timestamp with +00:00", ts2 is not None and ts2.year == 2026)

    check("parse_timestamp None", parse_timestamp(None) is None)
    check("parse_timestamp empty", parse_timestamp("") is None)

    # Test 2: extract_commit_message
    msg = extract_commit_message('git commit -m "Fix the bug"')
    check("extract_commit_message simple", msg == "Fix the bug", f"got: {msg}")

    msg2 = extract_commit_message(
        "git commit -m \"$(cat <<'EOF'\nAdd feature\n\nDetails\nEOF\n)\""
    )
    check("extract_commit_message heredoc", msg2 == "Add feature", f"got: {msg2}")

    check("extract_commit_message None", extract_commit_message(None) is None)
    check("extract_commit_message no git", extract_commit_message("ls -la") is None)

    # Test 3: parse synthetic JSONL
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "sessionId": "test-1234-5678",
                    "gitBranch": "test-branch",
                    "timestamp": "2026-01-01T10:00:00.000Z",
                    "message": {"role": "user", "content": "hello"},
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T10:00:05.000Z",
                    "message": {
                        "model": "claude-opus-4-6",
                        "role": "assistant",
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 200,
                            "cache_creation_input_tokens": 50,
                            "cache_read_input_tokens": 25,
                        },
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": "ls"},
                            },
                            {
                                "type": "tool_use",
                                "name": "Read",
                                "input": {"file_path": "/tmp/test"},
                            },
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T10:01:00.000Z",
                    "message": {
                        "model": "claude-opus-4-6",
                        "role": "assistant",
                        "usage": {
                            "input_tokens": 150,
                            "output_tokens": 300,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 100,
                        },
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": 'git commit -m "Test commit"'},
                            },
                            {
                                "type": "tool_use",
                                "name": "Skill",
                                "input": {"skill": "tdd:ci"},
                            },
                            {
                                "type": "tool_use",
                                "name": "Task",
                                "input": {
                                    "subagent_type": "Explore",
                                    "description": "Research something",
                                    "prompt": "...",
                                },
                            },
                        ],
                    },
                }
            ),
        ]
        tmp.write("\n".join(lines) + "\n")
        tmp.flush()
        tmp_path = Path(tmp.name)

    try:
        stats = parse_session_jsonl(tmp_path)

        check(
            "session_id extracted",
            stats["session_id"] == "test-1234-5678",
            f"got: {stats['session_id']}",
        )
        check(
            "branch extracted",
            stats["branch"] == "test-branch",
            f"got: {stats['branch']}",
        )
        check(
            "duration computed",
            stats["duration_minutes"] == 1,
            f"got: {stats['duration_minutes']}",
        )

        opus = stats["models"].get("claude-opus-4-6", {})
        check(
            "opus messages",
            opus.get("messages") == 2,
            f"got: {opus.get('messages')}",
        )
        check(
            "opus input_tokens",
            opus.get("input_tokens") == 250,
            f"got: {opus.get('input_tokens')}",
        )
        check(
            "opus output_tokens",
            opus.get("output_tokens") == 500,
            f"got: {opus.get('output_tokens')}",
        )
        check(
            "opus cache_creation_tokens",
            opus.get("cache_creation_tokens") == 50,
            f"got: {opus.get('cache_creation_tokens')}",
        )
        check(
            "opus cache_read_tokens",
            opus.get("cache_read_tokens") == 125,
            f"got: {opus.get('cache_read_tokens')}",
        )

        check(
            "total input",
            stats["total_tokens"]["input"] == 250,
            f"got: {stats['total_tokens']['input']}",
        )
        check(
            "total output",
            stats["total_tokens"]["output"] == 500,
            f"got: {stats['total_tokens']['output']}",
        )

        check(
            "cost > 0",
            stats["estimated_cost_usd"] > 0,
            f"got: {stats['estimated_cost_usd']}",
        )

        # Verify cost calculation manually:
        # 250 * 15/1M + 500 * 75/1M + 50 * 18.75/1M + 125 * 1.50/1M
        # = 0.00375 + 0.0375 + 0.0009375 + 0.0001875 = 0.042375
        expected_cost = round(
            250 * 15 / 1_000_000
            + 500 * 75 / 1_000_000
            + 50 * 18.75 / 1_000_000
            + 125 * 1.50 / 1_000_000,
            2,
        )
        check(
            "cost calculation",
            stats["estimated_cost_usd"] == expected_cost,
            f"expected: {expected_cost}, got: {stats['estimated_cost_usd']}",
        )

        check(
            "Bash count",
            stats["tools"].get("Bash") == 2,
            f"got: {stats['tools'].get('Bash')}",
        )
        check(
            "Read count",
            stats["tools"].get("Read") == 1,
            f"got: {stats['tools'].get('Read')}",
        )
        check(
            "Skill count",
            stats["tools"].get("Skill") == 1,
            f"got: {stats['tools'].get('Skill')}",
        )
        check(
            "Task count",
            stats["tools"].get("Task") == 1,
            f"got: {stats['tools'].get('Task')}",
        )

        check(
            "skills_invoked",
            len(stats["skills_invoked"]) == 1
            and stats["skills_invoked"][0]["skill"] == "tdd:ci",
            f"got: {stats['skills_invoked']}",
        )

        check(
            "subagents launched",
            len(stats["subagents"]) == 1 and stats["subagents"][0]["type"] == "Explore",
            f"got: {stats['subagents']}",
        )

        check(
            "commits found",
            len(stats["commits"]) == 1
            and stats["commits"][0]["message"] == "Test commit",
            f"got: {stats['commits']}",
        )

        check(
            "session_id_short",
            stats["session_id_short"] == "test-123",
            f"got: {stats['session_id_short']}",
        )

        check("problems_faced is list", isinstance(stats["problems_faced"], list))
        check("workflow_edges is dict", isinstance(stats["workflow_edges"], dict))
    finally:
        tmp_path.unlink(missing_ok=True)

    # Summary
    total = passed + failed
    print(f"\nSelf-test: {passed}/{total} passed", file=sys.stderr)
    if failed > 0:
        sys.exit(1)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.phase:
        print("ERROR: --phase is required (or use --self-test)", file=sys.stderr)
        sys.exit(1)

    if args.phase == "stats":
        phase_stats(args)
    elif args.phase in ("mermaid", "comment", "summary", "extract", "dashboard"):
        phase_placeholder(args.phase, args)
    else:
        print(f"ERROR: unknown phase '{args.phase}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
