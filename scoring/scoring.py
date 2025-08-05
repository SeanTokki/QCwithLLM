from langgraph.graph.state import CompiledStateGraph

from scoring.graphs.category_graph import run_graph as run_cat_graph
from scoring.graphs.image_graph import run_graph as run_img_graph
from scoring.graphs.additional_graph import run_graph as run_add_graph
from schema import *

async def get_category_result(graph: CompiledStateGraph, store_data: Dict[str, Any]) -> Optional[CategoryResult]:
    try:
        category_result = await run_cat_graph(graph, store_data)
    except Exception as e:
        print(f"[메뉴 점수] 오류 발생: {e}")
        return None
    
    return category_result

async def get_image_result(graph: CompiledStateGraph, store_data: Dict[str, Any]) -> Optional[ImageResult]:
    try:
        image_result = await run_img_graph(graph, store_data)
    except Exception as e:
        print(f"[내외부 점수] 오류 발생: {e}")
        return None
    
    return image_result

async def get_additional_result(graph: CompiledStateGraph, store_data: Dict[str, Any]) -> Optional[AdditionalResult]:
    try:
        additional_result = await run_add_graph(graph, store_data)
    except Exception as e:
        print(f"[추가 점수] 오류 발생: {e}")
        return None
    
    return additional_result