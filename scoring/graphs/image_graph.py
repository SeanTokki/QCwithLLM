from langgraph.graph import StateGraph, START, END

from schema import *
from scoring.nodes.image_nodes import *

def build_graph():
    workflow = StateGraph(ImgGraphState)
    
    workflow.add_node("prompter", prompter_node)
    workflow.add_node("captioner", captioner_node)
    workflow.add_node("scorer", scorer_node)
    workflow.add_node("formatter", formatter_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("postprocessor", postprocessor_node)
    
    workflow.add_edge(START, "prompter")
    workflow.add_conditional_edges(
        "prompter",
        lambda s: s.branch,
        {
            "no_image": END,
            "success": "captioner"
        }
    )
    workflow.add_edge("captioner", "scorer")
    workflow.add_conditional_edges(
        "scorer",
        lambda s: s.branch,
        {
            "too_many_attempts": END,
            "success": "formatter"
        }
    )
    workflow.add_edge("formatter", "validator")
    workflow.add_conditional_edges(
        "validator",
        lambda s: s.branch,
        {
            "invalid": "scorer",
            "valid": "postprocessor"
        }
    )
    workflow.add_edge("postprocessor", END)
    
    return workflow.compile()

async def run_graph(graph: CompiledStateGraph, store_data: Dict[str, Any]) -> Optional[ImageResult]:
    # 시스템 프롬프트
    system_prompt = """
    ## Role
    당신은 매장 분석 전문가입니다.
    당신은 특정 매장이 제휴를 맺기 적합한 매장인지 판단하기 위해 점수 기준표에 따라 매장의 이미지를 보고 점수를 부여하는 역할을 맡고 있습니다.
    차근차근 단계별로 생각하며 지시를 수행하세요.
    """
    
    inputs = {
        "raw_store_data": store_data,
        "messages": [{"role": "system", "content": system_prompt}],
    }
    
    # 그래프 실행
    result= await graph.ainvoke(input=inputs)
 
    return result["image_result"]

def visualize_graph(graph_obj: CompiledStateGraph, file_path: str) -> bytes:
    g = graph_obj.get_graph()
    png_bytes = g.draw_mermaid_png(output_file_path=file_path)
    
    return png_bytes