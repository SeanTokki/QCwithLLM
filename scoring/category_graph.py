from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from schema import *
from scoring.category_nodes import *

def build_graph():
    workflow = StateGraph(GraphState)
    
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
    first_user_prompt = f"""
    ## Instruction
    - 아래의 "Matching Rule"을 참고하여 입력으로 주어진 매장에 대해 카테고리 매칭 및 스코어링을 진행하기 전 주어진 입력이 충분한지 판단합니다.
    - 충분하다고 판단되었을 경우, 도구를 호출하지 않습니다.
    - 충분하지 않다고 판단되었을 경우, 적절한 도구를 호출합니다.
    - 판단에 대한 근거를 출력으로 제시합니다.
    
    ## Matching Rule
    {matching_rule}
    
    ## Input:
    - 매장 이름: {name}
    - 네이버 카테고리: {category}
    - 메뉴 정보: {menu_list}
    - 리뷰 정보: {review_list}
    """   
    second_user_prompt = f"""
    ## Instruction
    - "Matching Rule"을 참고하여 입력으로 주어진 매장에 대해 카테고리 매칭 및 스코어링을 진행합니다.
    - 먼저, 이 매장을 설명하는 가장 적합한 상위 카테고리를 한 개 선택합니다.
    - 선택된 상위 카테고리 안에서 "scoring_rules"의 첫번째 규칙을 보고 적절한 하위 카테고리가 있는지 확인합니다.
    - 만약 적절한 하위 카테고리가 없다고 판단되면 "scoring_rules"의 다음 규칙을 보고 적절한 하위 카테고리가 있는지 확인합니다.
    - 이렇게 계속 진행하다 "scoring_rules"의 마지막 규칙까지 확인했지만 적절한 하위 카테고리가 없다면 하위 카테고리를 ""(빈 문자열)로 설정합니다.
    - 상위/하위 카테고리 매칭을 하면서 했던 생각들을 요약해서 판단에 대한 근거를 작성합니다.
    - 매칭된 하위 카테고리에 해당하는 점수를 부여합니다. 만약 매칭된 하위 카테고리가 없다면 0점을 부여합니다.
    - 반드시 기준 표에 존재하는 카테고리에만 매칭하세요.
    
    ## Matching Rule
    {matching_rule}
    
    ## Examples
    ### Input: 
    - 매장 이름: {ex_name}
    - 네이버 카테고리: {ex_category}
    - 메뉴 정보: {ex_menu_list}
    - 리뷰 정보: {ex_review_list}
    
    ### Response:
    {ex_response}
    
    ### Input:
    - 매장 이름: {name}
    - 네이버 카테고리: {category}
    - 메뉴 정보: {menu_list}
    - 리뷰 정보: {review_list}
    
    ### Response:
    """
    
    inputs = {
        "raw_store_data": store_data,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
        ],
        "first_user_prompt": first_user_prompt,
        "second_user_prompt": second_user_prompt,
    }
    
    # 그래프 실행
    try:
        result= await graph.ainvoke(input=inputs)
    except Exception as e:
        print(f"[메뉴 점수] 오류 발생: {e}")
        return None

    return result["matching_result"]

    
    
    
    
