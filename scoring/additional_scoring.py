from langchain_google_genai import ChatGoogleGenerativeAI
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
from langchain.schema.runnable import RunnableLambda
import json

from scoring.schema import *

async def get_additional_result(store_data: Dict[str, Any]) -> Any:
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
    
    # 첫번째 LLM에게 넘길 프롬프트
    system_prompt = """
    ## Role
    당신은 매장 평가 전문가입니다.
    당신은 특정 매장이 제휴를 맺기 적합한 매장인지 판단하기 위해 점수 기준표에 따라 특정 매장에 점수를 부여하는 역할을 맡고 있습니다.
    차근차근 단계별로 생각하며 지시를 수행하세요.
    """
    user_prompt = """
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
    user_prompt = user_prompt.format(
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

    # 첫번째 LLM: 추가 점수 항목 확인 및 점수 부여
    async def first_llm_invoke(args) -> str:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            thinking_budget=1024,
        )
    
        text_response = await llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": args["system_prompt"]
                },
                {
                    "role": "user",
                    "content": args["user_prompt"]
                }
            ],
            tools=[GenAITool(google_search={})],
        )
        
        return text_response.content
    
    # 하나의 chain으로 묶기 위해 함수로 감싸기
    first_llm = RunnableLambda(first_llm_invoke)
    
    # 두번째 LLM: 추가 점수 스코어링 결과를 formatting
    structured_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        thinking_budget=0,
    ).with_structured_output(AdditionalResultLLM)

    chain = first_llm | structured_llm


    try:
        response: AdditionalResultLLM = await chain.ainvoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
    except Exception as e:
        print(f"[추가 점수] 오류 발생: {e}")
        return None
        
    # 추가 점수 항목이 목록에 없으면 None 반환
    for add_item in response.items:
        if add_item.item not in score_map:
            return None
        # 항목과 점수 매칭 수동으로 진행
        add_item.score = score_map[add_item.item]
    
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
    else:
        additional_result = None
    
    return additional_result