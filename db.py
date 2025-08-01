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

def save_store(naver_id: str, data: dict):
    d = _load("stores.json")
    d[naver_id] = data
    _save("stores.json", d)

def get_score(naver_id: str) -> Optional[dict]:    
    return _load("scores.json").get(naver_id)

def save_score(naver_id: str, data: dict):
    d = _load("scores.json")
    d[naver_id] = data
    _save("scores.json", d)

def new_task(naver_id: str = None, query: str = None) -> Optional[str]:
    # naver_id나 query 둘 중 하나라도 있어야 함
    if not naver_id and not query:
        return None
    
    tid = str(uuid.uuid4())
    
    task = _load("tasks.json")
    now = datetime.now().isoformat(timespec="seconds")
    task[tid] = {
        "query": query, 
        "naver_id": naver_id, 
        "status": "PENDING",
        "progress": 0, 
        "started": now, 
        "updated": now
    }
    _save("tasks.json", task)
    
    if naver_id:
        add_running_task(naver_id, tid)
    else:
        add_running_task(query, tid)
    
    return tid

def update_task(tid: str, **patch: dict):
    task = _load("tasks.json")
    task[tid].update(patch)
    task[tid]["updated"] = datetime.now().isoformat(timespec="seconds")
    _save("tasks.json", task)

def get_task(tid: str) -> Optional[dict]: 
    return _load("tasks.json").get(tid)

def get_running_tid(key: str) -> Optional[str]: 
    running = _load("running.json")
    return running.get(key)

def add_running_task(key: str, tid: str):
    running = _load("running.json")
    running[key] = tid
    _save("running.json", running)

def delete_running_task(key: str):
    running = _load("running.json")
    running.pop(key, None)
    _save("running.json", running)