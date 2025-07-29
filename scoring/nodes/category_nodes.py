from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
import chromadb
import json

from schema import *
from utils.build_store_vector_db import build_plain_text

# LLM에게 전달할 더미 유사 매장 검색 도구
@tool
def similar_partner_search()-> List[Dict]:
    """
    기존 제휴 매장 중에서 현재 스코어링 중인 매장과 유사한 매장의 카테고리 매칭 결과를 최대 3개 반환

    Returns:
        List[Dict]: 검색된 제휴 매장 매칭 결과의 리스트
    """
    pass

# 진짜 실행시킬 유사 매장 검색 도구
def real_similar_partner_search(store_data: str)-> List[Dict]:
    client     = chromadb.PersistentClient(path="temp_chroma_store")
    coll       = client.get_or_create_collection("stores")
    
    query_text = build_plain_text(store_data)
    
    res = coll.query(
        query_texts = [f"query: {query_text}"],
        n_results   = 3,
        include     = ["distances", "metadatas"],
    )

    result_list = []
    for rank, (m, d) in enumerate(zip(res["metadatas"][0], res["distances"][0]), 1):
        result_list.append({"rank": rank, "distance": d, **m})
    
    return result_list

# 사용 가능한 tools
tools = [similar_partner_search]

# 프롬프트를 생성하는 노드
def prompter_node(state: ImgGraphState) -> Dict[str, Any]:
    store_data = state.raw_store_data
    
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
    
    return {
        "first_user_prompt": first_user_prompt, 
        "second_user_prompt": second_user_prompt
    }

# 첫번째 LLM을 실행시키는 노드
async def tool_checker_node(state: CatGraphState):
    # 첫번째 LLM: 주어진 정보만으로 카테고리 매칭이 가능한지 판단
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        thinking_budget=1024,
    ).bind_tools(tools)
    
    user_message = {"role": "user", "content": state.first_user_prompt}
    response = await llm.ainvoke(state.messages + [user_message])
    
    # tool call message 존재 여부에 따라 분기
    if response.tool_calls:
        branch = "tool_called"
    else:
        branch = "tool_not_called"
    
    return {"messages": [user_message, response], "branch": branch}

# state.messages[-1]을 보고 자동으로 tool을 실행해주는 노드
def tool_node(state: CatGraphState):
    last_message = state.messages[-1]
    
    outputs = []
    for tool_call in last_message.tool_calls:
        # similar_partner_search 도구가 호출된 경우 직접 store_data를 전달
        if tool_call["name"] == "similar_partner_search":
            tool_result = real_similar_partner_search(state.raw_store_data)
        else:
            tool_result = tools[tool_call["name"]].invoke(tool_call["args"])

    outputs.append(
        ToolMessage(
            content=json.dumps(
                tool_result, ensure_ascii=False
            ),
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        )
    )
    
    return {"messages": outputs}

# 두번째 LLM을 실행시키는 노드
async def scorer_node(state: CatGraphState):
    # 3회 이상 재시도시 workflow 종료
    if state.attempts > 3:
        return {"matching_result": None, "branch": "too_many_attempts"}
    
    # 두번째 LLM: 카테고리 매칭 및 스코어링을 진행하고 format에 맞게 반환
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        thinking_budget=1024,
    ).with_structured_output(CategoryResult)
    
    user_message = {"role": "user", "content": state.second_user_prompt}
    response = await llm.ainvoke(state.messages + [user_message])
    ai_message = {"role": "ai", "content": response.model_dump_json()}
    
    return {"messages": [user_message, ai_message], "matching_result": response, "branch": "success"}

# 카테고리 매칭 결과의 유효성을 체크하는 노드
def validator_node(state: CatGraphState):
    with open("./data/rules/category_score_rule.json", "r", encoding="utf-8") as f:
        matching_rule = json.load(f)
    
    rule_map: dict[str, dict[str, int]] = {}
    for cat in matching_rule:
        top = cat["top_category"]
        sub_map: dict[str, int] = {}
        for block in cat["scoring_rules"]:
            for score_str, subs in block.items():
                score = int(score_str)
                for sub in subs:
                    sub_map[sub] = score
        sub_map[""] = 0 # 미분류 케이스 추가
        rule_map[top] = sub_map
    
    top = state.matching_result.top_category
    sub = state.matching_result.sub_category
    score = state.matching_result.score
    
    retry_reason = """
    카테고리 매칭 결과가 잘못되었으므로 재매칭하세요.
    잘못된 이유: 
    """
    branch = "valid"
    
    if not isinstance(score, int) or not 0 <= score <= 5:
        branch = "invalid"
        retry_reason += "score는 0부터 5까지의 정수여야 합니다.\n"

    if top not in rule_map:
        branch = "invalid"
        retry_reason += f"'{top}' 상위 카테고리가 매칭 목록에 존재하지 않습니다.\n"

    else:
        sub_scores = rule_map[top]

        if sub not in sub_scores:
            branch = "invalid"
            retry_reason += f"'{sub}' 하위 카테고리가 '{top}' 상위 카테고리의 목록에 존재하지 않습니다.\n"

        else:
            expected = sub_scores[sub]
            if expected != score:
                branch = "invalid"
                retry_reason += f" '{sub}' 하위 카테고리에 대한 점수는 {expected}점 입니다.\n"

    if branch == "invalid":
        return {
            "branch": branch,
            "attempts": state.attempts + 1,
            "messages": [{"role": "user", "content": retry_reason}]
        }
    
    else:
        return {
            "branch": branch
        }

#=====================================================================================================
# 여기부터는 현재 사용 안함
#=====================================================================================================

def save_result(result: CategoryResult, store_data: Dict):
    # 로컬 json 파일에 매칭 결과 저장
    with open("./data/final/scored_new_store_data.json", "r", encoding="utf-8") as f:
        scored_list = json.load(f)
    
    rec = {**store_data, **result.model_dump()}
    scored_list.append(rec)
    
    with open("./data/final/scored_new_store_data.json", "w", encoding="utf-8") as f:
        json.dump(scored_list, f, ensure_ascii=False, indent=4) 
    
    # ChromaDB에 매칭 결과 저장
    # client     = chromadb.PersistentClient(path="chroma_store")
    # coll       = client.get_or_create_collection("stores")
    
    # ids = [str(rec["naver_id"])]
    # documents = [build_plain_text(rec)]
    # metadatas = [
    #     {
    #         "naver_id": rec["naver_id"], 
    #         "name": rec["name"], 
    #         "top_category": rec["top_category"],
    #         "sub_category": rec["sub_category"],
    #         "score": rec["score"],
    #         "reasoning": rec["reasoning"]
    #     }
    # ]
    
    # coll.upsert(ids=ids, metadatas=metadatas, documents=documents)
    
    # print("매칭 결과 저장 완료")

def human_node(state: CatGraphState):
    if state.ok:
        save_result(state.matching_result, state.raw_store_data)
        return {
            "branch": "good"
        }
    else:
        message = {
            "role": "user",
            "content": f"다음 피드백을 보고 재시도하세요.\n유저 피드백: {state.feedback}"
        }
        return {
            "branch": "bad",
            "attempts": state.attempts + 1,
            "messages": [message]
        }