from langgraph.graph import StateGraph, START, END

from schema import *
from scoring.nodes.image_nodes import *

def build_graph():
    workflow = StateGraph(ImgGraphState)
    
    workflow.add_node("preprocessor", preprocessor_node)
    workflow.add_node("dispatcher", dispatcher_node)
    workflow.add_node("inn_scorer", inn_scorer_node)
    workflow.add_node("seat_scorer", seat_scorer_node)
    workflow.add_node("formatter", formatter_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("postprocessor", postprocessor_node)
    
    workflow.add_edge(START, "preprocessor")
    workflow.add_conditional_edges(
        "preprocessor",
        lambda s: s.branch,
        {
            "no_image": END,
            "success": "dispatcher"
        }
    )
    workflow.add_edge("dispatcher", "inn_scorer")
    workflow.add_edge("dispatcher", "seat_scorer")
    workflow.add_edge("inn_scorer", "formatter")
    workflow.add_edge("seat_scorer", "formatter")
    workflow.add_edge("formatter", "validator")
    workflow.add_conditional_edges(
        "validator",
        lambda s: s.branch,
        {
            "valid": "postprocessor",
            "invalid inn": "inn_scorer",
            "invalid seat": "seat_scorer",
            "invalid both": "dispatcher",
            "too many attempts": END
        }
    )
    workflow.add_edge("postprocessor", END)
    
    return workflow.compile()

async def run_graph(graph: CompiledStateGraph, store_data: Dict[str, Any]) -> Optional[ImageResult]:
    # 시스템 프롬프트
    with open("./scoring/prompts/image_system.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    inputs = {
        "raw_store_data": store_data,
        "messages": [{"role": "system", "content": system_prompt}],
    }
    
    # 그래프 이미지 저장
    visualize_graph(graph, "./data/graphs/image_graph.png")
    
    # 그래프 실행
    result= await graph.ainvoke(input=inputs)
 
    return result["image_result"]

def visualize_graph(graph_obj: CompiledStateGraph, file_path: str) -> bytes:
    g = graph_obj.get_graph()
    png_bytes = g.draw_mermaid_png(output_file_path=file_path)
    
    return png_bytes