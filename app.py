from fastapi import FastAPI, Response, BackgroundTasks, HTTPException
import asyncio
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from datetime import datetime

from schema import *
from scoring.pipeline import run_full_pipeline, make_full_result
from scoring.category_graph import build_graph
from db import *

@asynccontextmanager
async def lifespan(app: FastAPI):
    # LLM 호출을 위한 환경 변수 설정
    load_dotenv(".env")
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    # 메뉴 점수 부여를 위한 그래프 빌드
    app.state.graph = await asyncio.to_thread(build_graph)

    yield
    
    # app 종료 후 해야 할 일 있다면 추가
    pass


app = FastAPI(
    title="Store Scoring API",
    lifespan=lifespan,
)


# Scoring 작업을 백그라운드에서 실행하는 함수
async def scoring_task(task_id: str, naver_id: str, graph: CompiledStateGraph):
    done = asyncio.Event()
    
    # 진행률을 증가시키는 비동기 함수
    async def increase_progress():
        progress = 0
        delta = 10
        while not done.is_set() and progress < 95:
            # 1초마다 진행률 증가
            await asyncio.sleep(1)
            if done.is_set():
                break
            progress += delta
            # 진행률 증가량 감소 (약 30초 후 96% 도달)
            delta = max(2, delta-1)
            update_task(task_id, progress=progress)

        return
    
    # 진행률 증가 작업을 백그라운드로 실행
    ip = asyncio.create_task(increase_progress())
    
    try:
        # 진행 중 상태로 업데이트
        update_task(task_id, status="RUNNING")

        # 매장 데이터 가져오기
        store = get_store(naver_id)
        if store is None:
            raise RuntimeError("Store not found")
        
        # 스코어링 파이프라인 실행
        result = await run_full_pipeline(graph, store)       
        full_result = make_full_result(naver_id, result)
        if full_result is None:
            raise RuntimeError("Scoring failed, incomplete results")
        save_score(naver_id, full_result.model_dump())
        
        # 진행률 증가 작업 중지 신호
        done.set() 
        # 진행률 증가 작업이 완료될 때까지 대기
        await ip
        
        # 작업 완료 상태로 업데이트
        update_task(task_id, status="SUCCESS", progress=100)
        
    except Exception as e:
        update_task(task_id, status="FAILURE", error=str(e)[:200])
        
    finally:
        # 실행중인 작업 목록에서 task 제거
        finish_running_task_with_nid(naver_id)
        # 진행률 증가 작업 중지 신호
        done.set() 
        # 진행률 증가 작업이 완료될 때까지 대기
        await ip 


@app.get("/stores/{naver_id}/score")
async def get_store_score(naver_id: str, response: Response, bg: BackgroundTasks):
    # 이미 점수가 계산되어 있다면 캐시된 결과를 반환
    cached = get_score(naver_id)
    if cached:
        return {"ready": True, "score": cached}
    
    # 매장 데이터가 없으면 404 에러
    if get_store(naver_id) is None:
        raise HTTPException(404, "store not found")
    
    task_id = get_running_tid_with_nid(naver_id)
    
    # 해당 매장에 대해 진행중인 작업이 없다면 백그라운드 작업으로 점수 계산 시작
    if not task_id:
        task_id = new_task(naver_id)
        bg.add_task(scoring_task, task_id, naver_id, app.state.graph)
    
    # 점수 계산 작업을 처리 중임을 알림
    response.status_code = 202
    return {"ready": False, "task_id": task_id}


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = get_task(task_id)
    
    if not task:
        raise HTTPException(404, "task not found")
    
    return {
        "status": task["status"],
        "progress": task["progress"],
        "updated": task["updated"]
    }
    



    
