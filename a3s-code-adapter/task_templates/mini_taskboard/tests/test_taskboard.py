import json
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from taskboard import load_tasks, save_tasks, cmd_add, cmd_list, cmd_done

def test_add_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text("[]")

    import taskboard
    monkeypatch.setattr(taskboard, "TASKS_FILE", tasks_file)

    cmd_add("Test task")
    tasks = load_tasks()
    assert len(tasks) == 1
    assert tasks[0]["title"] == "Test task"

def test_list_empty(capsys):
    cmd_list()
    captured = capsys.readouterr()
    assert "No tasks" in captured.out
