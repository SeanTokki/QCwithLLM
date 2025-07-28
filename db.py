# DB 연동 전에 간단히 테스트 용으로 json 파일에서 데이터를 불러오고 저장
import json, uuid, pathlib
from datetime import datetime
from typing import Optional

DATA_DIR = pathlib.Path("./data/test")

def _load(fname: str) -> Optional[dict]:
    file = DATA_DIR / fname
    
    if not file.exists():
        return {}
    
    with file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def _save(fname: str, data: dict):
    file = DATA_DIR / fname
    
    with file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_store(naver_id: str) -> Optional[dict]:      
    return _load("stores.json").get(naver_id)

def get_score(naver_id: str) -> Optional[dict]:    
    return _load("scores.json").get(naver_id)

def save_score(naver_id: str, data: dict):
    d = _load("scores.json")
    d[naver_id] = data
    _save("scores.json", d)

def new_task(naver_id: str) -> str:
    tid = str(uuid.uuid4())
    
    task = _load("tasks.json")
    now = datetime.now().isoformat(timespec="seconds")
    task[tid] = {"naver_id": naver_id, "status": "PENDING",
                 "progress": 0, "started": now, "updated": now}
    _save("tasks.json", task)
    
    running = _load("running.json")
    running[naver_id] = tid
    _save("running.json", running)
    
    return tid

def update_task(tid: str, **patch: dict):
    task = _load("tasks.json")
    task[tid].update(patch)
    task[tid]["updated"] = datetime.now().isoformat(timespec="seconds")
    _save("tasks.json", task)

def get_task(tid: str) -> Optional[dict]: 
    return _load("tasks.json").get(tid)

def get_running_tid_with_nid(naver_id: str) -> Optional[str]: 
    running = _load("running.json")
    return running.get(naver_id)


def finish_running_task_with_nid(naver_id: str):
    running = _load("running.json")
    running.pop(naver_id, None)
    _save("running.json", running)