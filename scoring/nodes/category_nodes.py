from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
import chromadb
import json

from schema import *
from utils.build_store_vector_db import build_plain_text

# LLM에게 전달할 유사 매장 검색 도구
@tool
def similar_partner_search()-> List[Dict]:
    """
    기존 제휴 매장 중에서 현재 스코어링 중인 매장과 유사한 매장의 카테고리 매칭 결과를 최대 3개 반환

    Returns:
        List[Dict]: 검색된 제휴 매장 매칭 결과의 리스트
    """
    pass

# 진짜 실행시킬 유사 매장 검색 도구
def real_similar_partner_search(store_data: Dict)-> List[Dict]:
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

# LLM에게 전달할 음식 이미지 불러오는 도구
@tool
def get_food_images() -> List[Dict]:
    """
    매장의 음식 이미지 정보를 불러옵니다.

    Returns:
        List[Dict]: 음식 이미지 컨텐츠
    """
    pass

# 진짜 실행시킬 음식 이미지 불러오는 도구
def real_get_food_images(image_list: List[str]) -> List[Dict]:
    contents = [{"type": "text", "text": f"다음은 해당 스코어링 매장에 대한 {len(image_list)}개의 음식 이미지입니다."}]
    for image_url in image_list:
        contents.append({"type": "image", "source_type": "url", "url": image_url})

    return contents

# 사용 가능한 tools
tools = [similar_partner_search, get_food_images]

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
    with open("./scoring/prompts/category_1.txt", "r", encoding="utf-8") as f:
        template = f.read()
    first_user_prompt = template.format(
        matching_rule=matching_rule,
        name=name,
        category=category,
        menu_list=menu_list,
        review_list=review_list
    )
    
    with open("./scoring/prompts/category_2.txt", "r", encoding="utf-8") as f:
        template = f.read()
    second_user_prompt = template.format(
        matching_rule=matching_rule,
        ex_name=ex_name,
        ex_category=ex_category,
        ex_menu_list=ex_menu_list,
        ex_review_list=ex_review_list,
        ex_response=ex_response,
        name=name,
        category=category,
        menu_list=menu_list,
        review_list=review_list
    )
    
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
    
    tool_messages = []
    human_messages = []
    for tool_call in last_message.tool_calls:
        # similar_partner_search 도구가 호출된 경우 직접 store_data를 전달
        if tool_call["name"] == "similar_partner_search":
            tool_result = real_similar_partner_search(state.raw_store_data)
            content = json.dumps(tool_result, ensure_ascii=False)
        # get_food_images 도구가 호출된 경우 직접 store_data["image_list"]를 전달
        elif tool_call["name"] == "get_food_images":
            tool_result = real_get_food_images(state.raw_store_data.get("food_image_list"))
            if len(tool_result) == 1:
                content = "음식 이미지가 존재하지 않습니다."
            else:
                content = "음식 이미지를 불러왔습니다."
                human_messages.append({"role": "user", "content": tool_result})
        else:
            tool_result = tools[tool_call["name"]].invoke(tool_call["args"])
            content = json.dumps(tool_result, ensure_ascii=False)

        tool_messages.append(
            ToolMessage(
                content=content,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    
    return {"messages": tool_messages + human_messages}

# 두번째 LLM을 실행시키는 노드
async def scorer_node(state: CatGraphState):
    # 3회 이상 재시도시 workflow 종료
    if state.attempts > 3:
        print(f"[메뉴 점수] 재시도 횟수 초과로 스코어링 불가")
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