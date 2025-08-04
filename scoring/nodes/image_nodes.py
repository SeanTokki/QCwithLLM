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
    
    # 프롬프트 정의
    with open("./scoring/prompts/image.txt", "r", encoding="utf-8") as f:
        template = f.read()
    first_user_prompt = template.format(
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