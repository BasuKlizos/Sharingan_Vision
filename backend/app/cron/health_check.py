#!/usr/bin/env python3
"""
CLI tool to check interview_monitor service health.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


HEALTH_FILE = Path(
    os.environ.get(
        "AI_CAMERA_MONITOR_HEALTH_FILE",
        "/tmp/health.json"
    )
)


# File Handling

def read_health() -> Optional[Dict[str, Any]]:
    if not HEALTH_FILE.exists():
        return None

    try:
        content = HEALTH_FILE.read_text().strip()
        if not content:
            return None
        return json.loads(content)
    except (json.JSONDecodeError, OSError):
        return None



# Validation / Processing

def evaluate_health(health: Dict[str, Any], max_stale: int) -> Dict[str, Any]:
    health = health.copy()

    last_success = health.get("last_success_at")
    stale = False

    if last_success:
        try:
            success_time = datetime.fromisoformat(
                last_success.replace("Z", "+00:00")
            )
            age = (datetime.now(timezone.utc) - success_time).total_seconds()

            if age > max_stale:
                stale = True
                health["healthy"] = False
                health["stale_seconds"] = int(age)

        except (ValueError, TypeError):
            stale = True
            health["healthy"] = False
    else:
        if health.get("status") not in ("starting",):
            stale = True
            health["healthy"] = False

    health["stale"] = stale
    return health



# Formatting

def format_health(health: Dict[str, Any]) -> str:
    lines = [
        f"Service:    {health.get('service', 'unknown')}",
        f"Status:     {health.get('status', 'unknown')}",
        f"Healthy:    {'YES' if health.get('healthy') else 'NO'}",
        f"PID:        {health.get('pid', 'N/A')}",
        f"Uptime:     {health.get('uptime_seconds', 0):.0f}s",
        f"Runs:       {health.get('run_count', 0)} (errors: {health.get('error_count', 0)})",
        f"Processed:  {health.get('candidates_processed', 0)} candidates",
        f"Last Run:   {health.get('last_run_at', 'never')}",
        f"Last OK:    {health.get('last_success_at', 'never')}",
    ]

    if health.get("last_error"):
        lines.append(
            f"Last Error: {health.get('last_error')} ({health.get('last_error_at')})"
        )

    if health.get("stale"):
        lines.append(
            f"WARNING: Service is stale (>{health.get('stale_seconds', 0)}s)"
        )
        
    return "\n".join(lines)



# CLI Logic

def run_once(args) -> int:
    health = read_health()

    if health is None:
        output = {"error": "Health file not found", "healthy": False}

        if args.json:
            print(json.dumps(output))
        else:
            print("ERROR: Health file not found. Service may not be running.")

        return 2

    health = evaluate_health(health, args.max_stale)

    if args.json:
        print(json.dumps(health, indent=2))
    else:
        print(format_health(health))

    return 0 if health.get("healthy") else 1


def main():
    parser = argparse.ArgumentParser(description="Check interview_monitor health")

    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--watch", action="store_true", help="Watch mode (refresh every 5s)")
    parser.add_argument("--max-stale", type=int, default=7200)

    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                exit_code = run_once(args)
                print("\n" + "=" * 50 + "\n")
                time.sleep(5)
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        sys.exit(run_once(args))


if __name__ == "__main__":
    main()