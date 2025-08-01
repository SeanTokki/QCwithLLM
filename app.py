from fastapi import FastAPI, Response, BackgroundTasks, HTTPException
import asyncio
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from schema import *
from db import *
from utils.event import EventBridge
from scoring.pipeline import run_full_pipeline, make_full_result
from scoring.graphs.category_graph import build_graph as build_cat_graph
from scoring.graphs.image_graph import build_graph as build_img_graph
from scoring.graphs.additional_graph import build_graph as build_add_graph

@asynccontextmanager
async def lifespan(app: FastAPI):
    # LLM 호출을 위한 환경 변수 설정
    load_dotenv(".env")
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    # 앱 시작 시 그래프 빌드
    app.state.graphs = {}
    app.state.graphs["cat"] = await asyncio.to_thread(build_cat_graph)
    app.state.graphs["img"] = await asyncio.to_thread(build_img_graph)
    app.state.graphs["add"] = await asyncio.to_thread(build_add_graph)
    
    yield
    
    # app 종료 후 해야 할 일 있다면 추가
    pass


app = FastAPI(
    title="Store Scoring API",
    lifespan=lifespan,
)


# Scoring 작업을 백그라운드에서 실행하는 함수
async def scoring_task(task_id: str, graphs: Dict[str, CompiledStateGraph]):
    task = get_task(task_id)
    naver_id = task.get("naver_id")
    query = task.get("query")
    
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
        # 스코어링 진행 중 상태로 업데이트
        update_task(task_id, status="SCORING")

        # 매장 데이터 가져오기
        store = get_store(naver_id)
        if store is None:
            raise RuntimeError("Store not found")
        
        # 스코어링 파이프라인 실행
        result = await run_full_pipeline(graphs, store)       
        full_result = make_full_result(naver_id, result)
        if full_result is None:
            raise RuntimeError("Scoring failed")
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
        delete_running_task(naver_id)
        delete_running_task(query)
        # 진행률 증가 작업 중지 신호
        done.set() 
        # 진행률 증가 작업이 완료될 때까지 대기
        await ip 


# naver_id에 해당하는 단일 매장의 점수를 가져오는 API
@app.get("/stores/{naver_id}/score")
async def get_score_by_id(naver_id: str, response: Response, bg: BackgroundTasks):
    # 이미 점수가 계산되어 있다면 캐시된 결과를 반환
    cached = get_score(naver_id)
    if cached:
        return {"ready": True, "score": cached}
    
    # 매장 데이터가 없으면 404 에러
    if get_store(naver_id) is None:
        raise HTTPException(404, "store not found")
    
    # 이미 해당 naver_id로 진행 중인 작업이 있다면 그 작업의 task_id 반환
    task_id = get_running_tid(naver_id)
    
    # 진행 중인 작업이 없다면 점수 계산 작업을 시작
    if not task_id:
        task_id = new_task(naver_id=naver_id)
        bg.add_task(scoring_task, task_id, app.state.graphs)
    
    # 작업을 처리 중임을 알림
    response.status_code = 202
    return {"ready": False, "task_id": task_id}


# task_id로 작업 상태를 조회하는 API
@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = get_task(task_id)
    
    if not task:
        raise HTTPException(404, "task not found")
    
    return task


# query를 통해서 점수 생성 작업 실행 
@app.post("/tasks")
def make_task_with_query(payload: QueryTaskPayload, response: Response):
    # 이미 해당 query로 진행 중인 작업이 있다면 그 작업의 task_id 반환
    task_id = get_running_tid(payload.query)
    
    # 새로운 query로 작업 생성 및 실행
    if not task_id:
        task_id = new_task(query=payload.query)
    
    # 이벤트 발송
    event_bridge = EventBridge()
    event_bridge.send("find_nid_and_crawl", {"query": payload.query})
    
    response.status_code = 202
    return {"task_id": task_id}


# 크롤러에서 naver_id를 찾았을 때 발생하는 callback
@app.post("/internal/callback/nid-found")
async def process_after_nid(payload: NaverIDPayload, bg: BackgroundTasks):
    # query를 key로 갖는 task_id 미리 저장
    task_id = get_running_tid(payload.query)
    if not task_id:
        raise HTTPException(404, "No running task found for this query")
    
    naver_id = payload.naver_id
    
    # naver_id를 찾지 못했다면 작업 상태를 FAILURE로 업데이트
    if not naver_id:
        update_task(task_id, status="FAILURE", error="Naver ID not found")
        delete_running_task(payload.query)
    
    else:
        # 작업에 naver_id 정보 추가
        update_task(task_id, naver_id=naver_id)
        
        # 크롤링이 진행 중인 경우 작업 상태를 CRAWLING으로 업데이트
        # (이미 점수가 있더라도 크롤링을 진행하고 있다면 새로운 정보로 다시 스코어링 해야하므로)
        if payload.need_crawling:
            add_running_task(naver_id, task_id) # naver_id를 key로 running queue에 추가
            update_task(task_id, status="CRAWLING")
        
        # 크롤링 미진행 중이며 점수가 이미 있다면 작업 상태를 SUCCESS로 업데이트 후 종료
        elif get_score(naver_id):
            update_task(task_id, status="SUCCESS")
            delete_running_task(payload.query)
            return
        
        # 크롤링 미진행 중이며 점수가 없다면 점수 계산 작업을 시작
        else:
            add_running_task(naver_id, task_id) # naver_id를 key로 running queue에 추가
            bg.add_task(scoring_task, task_id, app.state.graphs)


# 크롤링 완료시 호출되는 callback
@app.post("/internal/callback/crawl-completed")
async def process_after_crawling(payload: CrawlCompPayload, bg: BackgroundTasks):
    naver_id = payload.naver_id
    task_id = get_running_tid(naver_id)
    query = get_task(task_id).get("query")
    
    if not task_id:
        raise HTTPException(404, "No running task found for this naver_id")
    
    # 크롤링이 성공적으로 완료되었으면 점수 계산 작업을 시작
    if payload.success:
        bg.add_task(scoring_task, task_id, app.state.graphs)

    # 크롤링이 실패했으면 작업 상태를 실패로 업데이트 후 실행 목록에서 제거
    else:
        update_task(task_id, status="FAILURE", error="Crawling failed")
        delete_running_task(naver_id)
        delete_running_task(query)
        
        
# =======================================================================================
# 여기서부터 테스트용 API
# =======================================================================================


# 매장 데이터 조회
@app.get("/internal/stores/{naver_id}")
def get_store_by_id(naver_id: str):
    data = get_store(naver_id)
    
    if data:
        return data
    else:
        raise HTTPException(404, "Store not found")


# 매장 데이터 저장
@app.post("/internal/stores/")
def save_store_with_payload(payload: Dict[str, Any]):
    naver_id = payload.get("naver_id")
    
    if naver_id:
        save_store(naver_id, payload)
        return {"msg": "saved", "naver_id": naver_id}
    else:
        raise HTTPException(422, "Data is wrong")