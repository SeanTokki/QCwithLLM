from langgraph.graph.state import CompiledStateGraph

from scoring.category_graph import run_graph
from schema import *

async def get_category_result(graph: CompiledStateGraph, store_data: Dict[str, Any]) -> Optional[CategoryResult]:
    try:
        category_result = await run_graph(graph, store_data)
    except Exception as e:
        print(f"[카테고리 점수] 오류 발생: {e}")
        return None
    
    return category_result
