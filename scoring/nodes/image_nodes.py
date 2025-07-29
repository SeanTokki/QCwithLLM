from langchain_google_genai import ChatGoogleGenerativeAI
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
import json

from schema import *

# 프롬프트를 생성하는 노드
def prompter_node(state: ImgGraphState) -> Dict[str, Any]:
    store_data = state.raw_store_data
    
    # 첫번째 LLM에 넘길 이미지 contents 생성
    if store_data["image_list"] or store_data["inner_image_list"]:
        image_list = store_data["image_list"] + store_data["inner_image_list"]
    else:
        print(f"[매장 내부 점수] 이미지가 없어서 스코어링 불가")
        return {"image_result": None, "branch": "no_image"}
    image_contents = []
    for image_url in image_list:
        image_contents.append({"type": "image", "source_type": "url", "url": image_url})
    
    # example용 데이터
    with open("./data/examples/example_scored_store_data.json", "r", encoding="utf-8") as f:
        ex_store_data = json.load(f)
    ex_response = ImageResultLLM(
        name=ex_store_data["name"],
        first_score=ex_store_data["inn_score"],
        first_reason=ex_store_data["inn_reason"],
        first_reason_images=[1, 4, 5],
        second_score=ex_store_data["seat_score"],
        second_reason=ex_store_data["seat_reason"],
        second_reason_images=[2, 3]
    )
    
    first_user_prompt = """
    ## Instruction
    - "Scoring Rule"을 참고하여 입력으로 주어진 매장의 이미지를 보고 스코어링을 진행합니다.
    - 주어진 매장 이미지의 인덱스는 1부터 시작합니다.
    - 1차 검수를 진행하여 점수를 부여하고 그 근거를 서술합니다.
    - 주어진 이미지 중 1차 검수 과정에서 큰 영향을 미친 이미지들을 골라 근거로써 첨부할 수 있습니다.
    - 1차 검수에서 주어진 이미지만으로 판단이 불가능할 경우 0점을 부여하고 그 이유를 서술합니다.
    - 2차 검수를 진행하여 점수를 부여하고 그 근거를 서술합니다.
    - 주어진 이미지 중 2차 검수 과정에서 큰 영향을 미친 이미지들을 골라 근거로써 첨부할 수 있습니다.
    - 2차 검수에서 주어진 이미지만으로 판단이 불가능할 경우, 검색 도구를 사용하여 추가 정보를 획득할 수 있습니다.
    - 추가 정보를 통해서도 판단이 불가능할 경우 0점을 부여하고 그 이유를 서술합니다.
    - 추가 정보를 통해 점수를 부여했다면 꼭 추가정보 내용과 출처를 근거에 포함시킵니다.

    ## Scoring Rule
    1차 검수: 내부 인테리어
    - 5점: 감각적 인테리어, 컨셉 인테리어(예시: 캠핑 컨셉으로 텐트와 캠핑의자, 한옥, 공장 개조형) 루프탑, 다락방, 오션뷰
    - 4점: 전체 특정 컨셉(마법당, 러빈허, 특정 나라 컨셉), 대형 쇼핑몰에 입점한 매장
    - 3점: 특정 컨셉 없는 통일된 톤의 인테리어(조명이 전체적으로 밝은 매장)
    - 2점: 특정 컨셉 없고, 통일되지 않는 톤으로 이루어진 인테리어의 매장
    - 1점: 그 외 매장
    2차 검수: 좌석 수에 따른 점수 부여
    - 30석 초과시 0.5점
    - 30석 이하 10석 초과시 0점
    - 10석 이하시 -0.5점

    ## Examples
    ### Input:
    '{ex_name}' 매장에 대해 평가를 진행하세요.

    ### Response:
    {ex_response}

    ### Input: 
    '{name}' 매장에 대해 평가를 진행하세요.
    """
    first_user_prompt = first_user_prompt.format(
        ex_name=ex_store_data["name"],
        ex_response=ex_response.model_dump(),
        name=state.raw_store_data["name"]
    )
    second_user_prompt = """
    ## Instruction
    - 이전의 응답을 주어진 형식에 맞게 변환합니다.
    """
    
    return {
        "first_user_prompt": first_user_prompt, 
        "second_user_prompt": second_user_prompt,
        "image_contents": image_contents,
        "branch": "success"
    }

# 첫번째 LLM을 실행시키는 노드
async def scorer_node(state: ImgGraphState) -> Dict[str, Any]:
    # 3회 이상 재시도시 workflow 종료
    if state.attempts > 3:
        return {"image_result": None, "branch": "too_many_attempts"}
    
    # 첫번째 LLM: 추가 점수 항목 확인 및 점수 부여
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        thinking_budget=1024,
    )
    
    user_message = {
        "role": "user", 
        "content": [{"type": "text", "text": state.first_user_prompt}] + state.image_contents
    }
    response = await llm.ainvoke(
        state.messages + [user_message], 
        tools=[GenAITool(google_search={})]
    )
    
    return {"messages": [user_message, response], "branch": "success"}

# 두번째 LLM을 실행시키는 노드
async def formatter_node(state: ImgGraphState) -> Dict[str, Any]:
    # 두번째 LLM: 추가 점수 스코어링 결과를 formatting
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        thinking_budget=0,
    ).with_structured_output(ImageResultLLM)
    
    user_message = {"role": "user", "content": state.second_user_prompt}
    response = await llm.ainvoke(state.messages + [user_message])
    ai_message = {"role": "ai", "content": response.model_dump_json()}
    
    return {"messages": [user_message, ai_message], "image_result": response}

# 이미지 평가 결과의 유효성을 체크하는 노드
def validator_node(state: ImgGraphState) -> Dict[str, Any]:
    response = state.image_result
    
    retry_reason = """
    평가 결과가 잘못되었으므로 재평가하세요.
    잘못된 이유: 
    """
    branch = "valid"
    
    if response.first_score not in [1.0, 2.0, 3.0, 4.0, 5.0]:
        branch = "invalid"
        retry_reason += "1차 점수는 1.0, 2.0, 3.0, 4.0, 5.0 중의 하나여야 합니다.\n"
    if response.second_score not in [0.0, 0.5, -0.5]:
        branch = "invalid"
        retry_reason += "2차 점수는 0.0, 0.5, -0.5 중의 하나여야 합니다.\n"
    
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
def postprocessor_node(state: ImgGraphState) -> Dict[str, Any]:
    response = state.image_result
    image_contents = state.image_contents
    
    # 인덱스로 주어진 image 정보를 url string으로 변환
    first_image_urls = []
    for image_url in response.first_reason_images:
        if image_url <= len(image_contents):
            first_image_urls.append(image_contents[image_url-1]["url"])
    second_image_urls = []
    for image_url in response.second_reason_images:
        if image_url <= len(image_contents):
            second_image_urls.append(image_contents[image_url-1]["url"])
    
    # 1차 점수 미부여시 2차 검수 미진행으로 처리
    if response.first_score == 0:
        response.second_score = 0
        response.second_reason = "1차 점수 미부여로 2차 검수를 진행하지 않습니다."
        second_image_url = None
    
    # 1차 점수와 2차 점수 합산으로 최종 점수 계산
    total_score = response.first_score + response.second_score
    
    image_result = ImageResult(
        name=response.name, 
        first_score=response.first_score,
        first_reason=response.first_reason,
        first_reason_images=first_image_urls,
        second_score=response.second_score,
        second_reason=response.second_reason,
        second_reason_images=second_image_urls,
        score=total_score
    )
    
    return {"image_result": image_result}