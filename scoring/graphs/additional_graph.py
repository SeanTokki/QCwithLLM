from langgraph.graph import StateGraph, START, END

from schema import *
from scoring.nodes.additional_nodes import *

def build_graph():
    workflow = StateGraph(AddGraphState)
    
    workflow.add_node("scorer", scorer_node)
    workflow.add_node("formatter", formatter_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("preprocessor", preprocessor_node)
    
    workflow.add_edge(START, "scorer")
    workflow.add_edge("scorer", "formatter")
    workflow.add_edge("formatter", "validator")
    workflow.add_conditional_edges(
        "validator",
        lambda s: s.branch,
        {
            "invalid": "scorer",
            "valid": "preprocessor"
        }
    )
    workflow.add_edge("preprocessor", END)
    
    return workflow.compile()

async def run_graph(graph: CompiledStateGraph, store_data: Dict[str, Any]) -> Optional[AdditionalResult]:
    # 추가 점수 부여 규칙 불러오기
    with open("./data/rules/additional_score_rule.json", "r", encoding="utf-8") as f:
        scoring_rule = json.load(f)
    score_map = {x["item"]: x["score"] for x in scoring_rule}
    
    # example용 데이터
    with open("./data/examples/example_scored_store_data.json", "r", encoding="utf-8") as f:
        ex_store_data = json.load(f)
    ex_response = AdditionalResultLLM(
        name=ex_store_data["name"],
        items=ex_store_data["add_items"]
    )
    
    # 프롬프트 정의
    system_prompt = """
    ## Role
    당신은 매장 평가 전문가입니다.
    당신은 특정 매장이 제휴를 맺기 적합한 매장인지 판단하기 위해 점수 기준표에 따라 특정 매장에 점수를 부여하는 역할을 맡고 있습니다.
    차근차근 단계별로 생각하며 지시를 수행하세요.
    """
    first_user_prompt = """
    ## Instruction
    - "Scoring Rule"과 "Additional Rule"을 참고하여 입력으로 주어진 매장의 정보를 활용해 모든 추가 점수 항목에 대해 이 매장이 해당하는지 확인합니다.
    - 주어진 정보만으로는 확인되지 않는 항목에 대해서는 검색을 통해서 알아볼 수 있습니다.
    - 각 항목에 대해 이 매장이 해당한다면 왜 해당하는지, 해당하지 않는다면 왜 해당하지 않는지 이유를 서술합니다.
    - 검색을 사용하여 얻은 정보를 통해 판단한 항목에 대해서는 반드시 검색 내용을 이유에 포함시킵니다.

    ## Scoring Rule
    {scoring_rule}
    
    ## Additional Rule
    - TV 방영 여부를 따질 때 유튜브는 포함되지 않습니다.
    - TV 방영의 조건은 해당 매장에 대한 언급이 있거나 해당 매장이 화면에 나왔어야 합니다.
    - 유명인/연예인은 누구나 알만한 사람이어야 합니다.
    - 신메뉴 출시는 주기적 상품 편동에 해당하지 않습니다.
    - 꼭 "주기적"으로 메뉴가 변하는 것이 확인되었을 경우에만 주기적 상품 변동에 해당합니다.
    - 일회용잔 카페는 "항상" 일회용품만 사용하는 "카페" 일때만 해당합니다.

    ## Examples
    ### Input:
    다음 매장에 대해 평가를 진행하세요.
    - 매장 이름: {ex_name}
    - 매장 주소: {ex_address}
    - 메뉴 정보: {ex_menu_list}
    - 리뷰 정보: {ex_review_list}
    
    ### Response:
    {ex_response}

    ### Input:
    다음 매장에 대해 평가를 진행하세요.
    - 매장 이름: {name}
    - 매장 주소: {address}
    - 메뉴 정보: {menu_list}
    - 리뷰 정보: {review_list}

    ### Response:
    """
    first_user_prompt = first_user_prompt.format(
        scoring_rule=scoring_rule,
        name=store_data["name"],
        address=store_data["address"],
        menu_list=store_data["menu_list"], 
        review_list=store_data["review_list"], 
        ex_name=ex_store_data["name"],
        ex_address=ex_store_data["address"],
        ex_menu_list=ex_store_data["menu_list"], 
        ex_review_list=ex_store_data["review_list"],
        ex_response=ex_response.model_dump()
    )
    second_user_prompt = """
    ## Instruction
    - 이전의 응답을 주어진 형식에 맞게 변환합니다.
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
    result= await graph.ainvoke(input=inputs)
 
    return result["additional_result"]

def visualize_graph(graph_obj: CompiledStateGraph, file_path: str) -> bytes:
    g = graph_obj.get_graph()
    png_bytes = g.draw_mermaid_png(output_file_path=file_path)
    
    return png_bytes