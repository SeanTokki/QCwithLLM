from dotenv import load_dotenv
import os
import asyncio
import json
from time import perf_counter
import random
from tqdm import tqdm
import csv

from scoring.category_scoring import get_category_result
from scoring.image_scoring import get_image_result
from scoring.additional_scoring import get_additional_result
from scoring.position_scoring import get_position_result
from scoring.category_graph import build_graph
from scoring.schema import *

class FullResult(BaseModel):
    naver_id: int
    name: str
    pos_score: float
    pos_reason: str
    top_cat: str
    sub_cat: str
    cat_reason: str
    cat_score: float
    inn_score: float
    inn_reason: str
    inn_reason_img: Optional[str]
    seat_score: float
    seat_reason: str
    seat_reason_img: Optional[str]
    img_score: float
    add_items: List[AdditionalItem]
    add_score: float
    tot_score: float
    

async def run_full_pipeline(graph, store_data):
    tasks = {
        "category_result": lambda: get_category_result(graph, store_data),
        "image_result": lambda: get_image_result(store_data),
        "additional_result": lambda: get_additional_result(store_data),
        "position_result": lambda: get_position_result(store_data)
    }
    
    result = {k: None for k in tasks.keys()}
    
    for _ in range(3):
        pending = [k for k, v in result.items() if v is None]
        if not pending:
            break

        new_values = await asyncio.gather(*[tasks[k]() for k in pending])
        result.update(dict(zip(pending, new_values)))
    
    return result

def calc_score(result):
    total_score = 0
    total_score += result["position_result"].score
    total_score += result["category_result"].score
    total_score += result["image_result"].score
    total_score /= 3
    total_score += result["additional_result"].score
    
    return round(total_score, 1)

def make_full_result(naver_id, result) -> Optional[FullResult]:
    pos_res: PositionResult = result["position_result"]
    cat_res: CategoryResult = result["category_result"]
    img_res: ImageResult = result["image_result"]
    add_res: AdditionalResult = result["additional_result"]
    
    if None in [pos_res, cat_res, img_res, add_res]:
        return None
    
    full_result = FullResult(
        naver_id=naver_id,
        name=pos_res.name,
        pos_score=pos_res.score,
        pos_reason=pos_res.reason,
        top_cat=cat_res.top_category,
        sub_cat=cat_res.sub_category,
        cat_reason=cat_res.reason,
        cat_score=cat_res.score,
        inn_score=img_res.first_score,
        inn_reason=img_res.first_reason,
        inn_reason_img=img_res.first_reason_image,
        seat_score=img_res.second_score,
        seat_reason=img_res.second_reason,
        seat_reason_img=img_res.second_reason_image,
        img_score=img_res.score,
        add_items=add_res.items,
        add_score=add_res.score,
        tot_score=calc_score(result)
    )
    
    return full_result

if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    graph = build_graph()
    with open("./data/test/store_data/merged_filtered_store_data.json", "r", encoding="utf-8") as f:
        store_data = json.load(f)
    data = store_data[0]
    
    result = asyncio.run(run_full_pipeline(graph, data))
    full_result = make_full_result(data["naver_id"], result)
    print(full_result)
    
    # for data in tqdm(store_data, "scoring"):
    #     result = asyncio.run(run_full_pipeline(graph, data))
    #     full_result = make_full_result(data["naver_id"], result)
    #     if full_result == None:
    #         continue

    #     with open("./data/for_request/add/full_scoring_result.json", "r", encoding="utf-8") as f:
    #         results = json.load(f)
        
    #     results.append(full_result.model_dump())
        
    #     with open("./data/for_request/add/full_scoring_result.json", "w", encoding="utf-8") as f:
    #         json.dump(results, f, ensure_ascii=False, indent=4)

