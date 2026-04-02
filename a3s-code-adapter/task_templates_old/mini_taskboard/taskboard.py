#!/usr/bin/env python3
"""Mini Taskboard - A simple CLI task manager"""
import json
import sys
from pathlib import Path

TASKS_FILE = Path(__file__).parent / "tasks.json"

def load_tasks():
    if not TASKS_FILE.exists():
        return []
    return json.loads(TASKS_FILE.read_text())

def save_tasks(tasks):
    TASKS_FILE.write_text(json.dumps(tasks, indent=2))

def cmd_add(title):
    tasks = load_tasks()
    task_id = max([t["id"] for t in tasks], default=0) + 1
    tasks.append({"id": task_id, "title": title, "done": False})
    save_tasks(tasks)
    print(f"Added task {task_id}: {title}")

def cmd_list():
    tasks = load_tasks()
    if not tasks:
        print("No tasks")
        return
    for t in tasks:
        status = "✓" if t["done"] else " "
        print(f"[{status}] {t['id']}: {t['title']}")

def cmd_done(task_id):
    tasks = load_tasks()
    for t in tasks:
        if t["id"] == int(task_id):
            t["done"] = True
            save_tasks(tasks)
            print(f"Marked task {task_id} as done")
            return
    print(f"Task {task_id} not found")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: taskboard.py [add|list|done] [args]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "add" and len(sys.argv) > 2:
        cmd_add(" ".join(sys.argv[2:]))
    elif cmd == "list":
        cmd_list()
    elif cmd == "done" and len(sys.argv) > 2:
        cmd_done(sys.argv[2])
    else:
        print("Invalid command")
