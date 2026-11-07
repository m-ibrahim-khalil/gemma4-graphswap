from pathlib import Path

from graphswap.tasks import commits, load_tasks

DATA = Path(__file__).resolve().parent.parent / "data"


def test_loads_127_distinct_commits():
    tasks = load_tasks(DATA / "tasks.jsonl")
    assert len(tasks) == 129
    assert len(commits(tasks)) == 127
    assert {t.repo_short for t in tasks} == {"fastapi", "rich", "requests", "httpx"}
