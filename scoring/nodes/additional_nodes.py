from langchain_google_genai import ChatGoogleGenerativeAI
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
import json

from schema import *

# 첫번째 LLM을 실행시키는 노드
async def scorer_node(state: AddGraphState) -> Dict[str, Any]:
    # 3회 이상 재시도시 workflow 종료
    if state.attempts > 3:
        return {"matching_result": None, "branch": "too_many_attempts"}
    
    # 첫번째 LLM: 추가 점수 항목 확인 및 점수 부여
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        thinking_budget=1024,
    )
    
    user_message = {"role": "user", "content": state.first_user_prompt}
    response = await llm.ainvoke(
        state.messages + [user_message], 
        tools=[GenAITool(google_search={})]
    )
    
    return {"messages": [user_message, response]}

# 두번째 LLM을 실행시키는 노드
async def formatter_node(state: AddGraphState) -> Dict[str, Any]:
    # 두번째 LLM: 추가 점수 스코어링 결과를 formatting
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        thinking_budget=0,
    ).with_structured_output(AdditionalResultLLM)
    
    user_message = {"role": "user", "content": state.second_user_prompt}
    response = await llm.ainvoke(state.messages + [user_message])
    ai_message = {"role": "ai", "content": response.model_dump_json()}
    
    return {"messages": [user_message, ai_message], "additional_result": response}

# 추가 점수 결과의 유효성을 체크하는 노드
def validator_node(state: AddGraphState) -> Dict[str, Any]:
    # 추가 점수 부여 규칙 불러오기
    with open("./data/rules/additional_score_rule.json", "r", encoding="utf-8") as f:
        scoring_rule = json.load(f)
    score_map = {x["item"]: x["score"] for x in scoring_rule}
    
    response = state.additional_result
    
    retry_reason = """
    다음 항목은 추가 점수 항목 목록에 없습니다.
    잘못된 항목: 
    """
    branch = "valid"
    
    for add_item in response.items:
        # 추가 점수 항목이 목록에 없으면 재시도
        if add_item.item not in score_map:
            retry_reason += add_item.item
            branch = "invalid"
        # 항목과 점수 매칭은 수동으로 진행
        add_item.score = score_map[add_item.item]
    
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

# LLM 결과에 이어서 추가적인 가공을 하는 노드
def preprocessor_node(state: AddGraphState) -> Dict[str, Any]:
    response = state.additional_result
    store_data = state.raw_store_data
    
    if response:
        # 미쉐린, 블로그 리뷰, 주차장 점수 추가
        response.items.append(
            AdditionalItem(
                item="미쉐린 가이드 등재", 
                score=0.5, 
                selected=True if store_data["seoul_michelin"] else False, 
                reason="수집 데이터에서 미쉐린 가이드 등재 여부를 찾았습니다."
            )
        )
        response.items.append(
            AdditionalItem(
                item="블로그 리뷰 300개", 
                score=0.3, 
                selected=(store_data["blog_review_count"] >= 300) if store_data["blog_review_count"] else False, 
                reason=f"수집 데이터에서 블로그 리뷰 수가 {store_data['blog_review_count']}개 입니다."
            )
        )
        response.items.append(
            AdditionalItem(
                item="자체 주차장 보유", 
                score=0.2, 
                selected=True if store_data["parking_available"] else False, 
                reason=f"수집 데이터에서 주차 가능 여부를 찾았습니다."
            )
        )
        
        # 해당하는 항목 점수 합산으로 최종 점수 계산
        # 해당하는 항목만 최종 결과물로 제출
        total_score = 0.0
        true_items = []
        for item in response.items:
            if item.selected:
                total_score += item.score
                true_items.append(item)
        total_score = round(total_score, 1)
        
        additional_result = AdditionalResult(
            name = response.name,
            items = true_items,
            score = total_score
        )
    
    return {"additional_result": additional_result}