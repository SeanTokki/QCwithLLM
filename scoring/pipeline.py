import asyncio

from scoring.scoring import *
from scoring.position_scoring import get_position_result
from schema import *

async def run_full_pipeline(graphs: Dict[str, CompiledStateGraph], store_data: Dict[str, Any]) -> Dict[str, Any]:
    tasks = {
        "category_result": lambda: get_category_result(graphs["cat"], store_data),
        "image_result": lambda: get_image_result(graphs["img"], store_data),
        "additional_result": lambda: get_additional_result(graphs["add"], store_data),
        "position_result": lambda: get_position_result(store_data)
    }
    
    result = {k: None for k in tasks.keys()}
    
    # 실패한 작업들만 최대 3회까지 실행
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
        inn_score=img_res.inn_score,
        inn_reason=img_res.inn_reason,
        inn_reason_idxs=img_res.inn_reason_idxs,
        inn_reason_imgs=img_res.inn_reason_images,
        seat_score=img_res.seat_score,
        seat_reason=img_res.seat_reason,
        seat_reason_idxs=img_res.seat_reason_idxs,
        seat_reason_imgs=img_res.seat_reason_images,
        img_score=img_res.score,
        add_items=add_res.items,
        add_score=add_res.score,
        tot_score=calc_score(result)
    )
    
    return full_result