from langgraph.graph import StateGraph, START, END

from schema import *
from scoring.nodes.category_nodes import *

def build_graph():
    workflow = StateGraph(CatGraphState)
    
    workflow.add_node("tool_checker", tool_checker_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("scorer", scorer_node)
    workflow.add_node("validator", validator_node)
    
    workflow.add_edge(START, "tool_checker")
    workflow.add_conditional_edges(
        "tool_checker", 
        lambda s: s.branch,
        {
            "tool_not_called": "scorer",
            "tool_called": "tools",
        }
    )
    workflow.add_edge("tools", "scorer")
    workflow.add_edge("scorer", "validator")
    workflow.add_conditional_edges(
        "validator",
        lambda s: s.branch,
        {
            "invalid": "scorer",
            "valid": END,
            "too_many_attempts": END
        }
    )
    
    return workflow.compile()

async def run_graph(graph: CompiledStateGraph, store_data: Dict[str, Any]) -> Optional[CategoryResult]:
    # 시스템 프롬프트
    with open("./scoring/prompts/category_system.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    inputs = {
        "raw_store_data": store_data,
        "messages": [{"role": "system", "content": system_prompt}],
    }
    
    # 그래프 이미지 저장
    visualize_graph(graph, "./data/graphs/category_graph.png")
    
    # 그래프 실행
    result= await graph.ainvoke(input=inputs)

    return result["matching_result"]

def visualize_graph(graph_obj: CompiledStateGraph, file_path: str) -> bytes:
    g = graph_obj.get_graph()
    png_bytes = g.draw_mermaid_png(output_file_path=file_path)
    
    return png_bytes
    
    
    
    
