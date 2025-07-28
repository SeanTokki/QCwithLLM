import asyncio

from scoring.category_scoring import get_category_result
from scoring.image_scoring import get_image_result
from scoring.additional_scoring import get_additional_result
from scoring.position_scoring import get_position_result
from schema import *

async def run_full_pipeline(graph: CompiledStateGraph, store_data: Dict[str, Any]) -> Dict[str, Any]:
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

def calc_score(result: Dict[str, Any]) -> float:
    total_score = 0
    total_score += result["position_result"].score
    total_score += result["category_result"].score
    total_score += result["image_result"].score
    total_score /= 3
    total_score += result["additional_result"].score
    
    return round(total_score, 1)

def make_full_result(naver_id: str, result: Dict[str, Any]) -> Optional[FullResult]:
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