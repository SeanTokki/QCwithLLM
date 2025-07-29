from langgraph.graph import StateGraph, START, END

from schema import *
from scoring.nodes.category_nodes import *

def build_graph():
    workflow = StateGraph(CatGraphState)
    
    workflow.add_node("prompter", prompter_node)
    workflow.add_node("tool_checker", tool_checker_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("scorer", scorer_node)
    workflow.add_node("validator", validator_node)
    
    workflow.add_edge(START, "prompter")
    workflow.add_edge("prompter", "tool_checker")
    workflow.add_conditional_edges(
        "tool_checker", 
        lambda s: s.branch,
        {
            "tool_not_called": "scorer",
            "tool_called": "tools",
        }
    )
    workflow.add_edge("tools", "scorer")
    workflow.add_conditional_edges(
        "scorer",
        lambda s: s.branch,
        {
            "too_many_attempts": END,
            "success": "validator"
        }
    )
    workflow.add_conditional_edges(
        "validator",
        lambda s: s.branch,
        {
            "invalid": "scorer",
            "valid": END
        }
    )
    
    return workflow.compile()

async def run_graph(graph: CompiledStateGraph, store_data: Dict[str, Any]) -> Optional[CategoryResult]:
    # 카테고리 매칭 규칙 불러오기
    with open("./data/rules/category_score_rule.json", "r", encoding="utf-8") as f:
        matching_rule = json.load(f)
    
    # example용 데이터
    with open("./data/examples/example_scored_store_data.json", "r", encoding="utf-8") as f:
        ex_store_data = json.load(f)
    
    target_keys = ["name", "category", "menu_list", "review_list"]
    name, category, menu_list, review_list = [store_data.get(k) for k in target_keys]
    ex_target_keys = ["name", "category", "menu_list", "review_list", "top_cat", "sub_cat", "cat_score", "cat_reason"]
    ex_name, ex_category, ex_menu_list, ex_review_list, ex_top_cat, ex_sub_cat, ex_score, ex_reason = [ex_store_data.get(k) for k in ex_target_keys]
    ex_response = json.dumps(
        {"name": ex_name, "top_category": ex_top_cat, "sub_category": ex_sub_cat, "score": ex_score, "reason": ex_reason},
        ensure_ascii=False,
    )
    
    # 프롬프트 정의
    system_prompt = """
    ## Role
    당신은 카테고리 매칭 전문가입니다.
    당신은 특정 매장이 제휴를 맺기 적합한 매장인지 판단하기 위해 점수 기준표에 따라 매장의 카테고리를 매칭하고 점수를 부여하는 역할을 맡고 있습니다.
    차근차근 단계별로 생각하며 지시를 수행하세요.
    """
    
    inputs = {
        "raw_store_data": store_data,
        "messages": [{"role": "system", "content": system_prompt}],
    }
    
    # 그래프 실행
    result= await graph.ainvoke(input=inputs)

    return result["matching_result"]

def visualize_graph(graph_obj: CompiledStateGraph, file_path: str) -> bytes:
    g = graph_obj.get_graph()
    png_bytes = g.draw_mermaid_png(output_file_path=file_path)
    
    return png_bytes
    
    
    
    
