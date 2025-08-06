from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
import json
import asyncio

from schema import *

# 프롬프트를 생성하는 노드
def prompter_node(state: ImgGraphState) -> Dict[str, Any]:
    store_data = state.raw_store_data
    
    # 첫번째 LLM에 넘길 내부 이미지 contents 생성
    if store_data["inner_image_list"]:
        image_list = store_data["inner_image_list"]
    else:
        print(f"[내외부 점수] 이미지가 없어서 스코어링 불가")
        return {"image_result": None, "branch": "no_image"}
    image_contents = []
    for image_url in image_list:
        image_contents.append({"type": "image", "source_type": "url", "url": image_url})
    
    # example용 가상 매장 데이터 (현재는 few-shot 없이 진행)
    ex_name = "브라운테이블"
    ex_captions = """
    0: 화이트 벽면과 밝은 우드 톤 가구가 어우러진 전형적인 카페 인테리어다. 4인용 테이블 4개(16석)가 균일하게 배치돼 있으며, 장식으로는 작은 화분 몇 개 정도만 보인다.
    1: 창가 쪽에는 통유리창이 있어 자연 채광이 들어오지만, 특수 소품이나 독특한 콘셉트 장식은 확인되지 않는다. 2인용 원형 테이블 3개(6석)가 추가로 놓여 있다.
    2: 카운터 벽면은 민트색 페인트로 포인트를 줬지만, 전체적으로는 우드·화이트 조합이 반복되는 평범한 디자인이다. 바 카운터 앞에 하이 스툴 4석이 보인다.
    3: 천장에는 레일 조명과 심플한 펜던트 등이 규칙적으로 설치돼 있어 무난한 카페 분위기를 유지한다. 별도 룸이나 파티션 같은 구획은 보이지 않는다.
    4: 매장 후면의 선반에는 머그잔과 소형 식물만 진열돼 있을 뿐, 테마성 소품·예술 작품은 없다. 사진에 잡힌 좌석은 총 26석으로 일반 중소형 카페 규모다.
    """
    ex_response = ImageResultLLM(
        name=ex_name,
        first_score=3,
        first_reason="캡션 0-4 모두에서 특별한 테마나 독창적 소품 없이 우드·화이트 톤의 일반적인 카페 인테리어만 확인됨.",
        first_reason_captions=[0, 1, 2, 3, 4],
        second_score=0.0,
        second_reason="캡션 0·1·2·4에서 확인된 좌석 합계가 26석으로 11-30석 구간에 해당하여 조정점수 0.",
        second_reason_captions=[0, 1, 2, 4]
    )
    # with open("./data/examples/example_scored_store_data.json", "r", encoding="utf-8") as f:
    #     ex_store_data = json.load(f)
    # ex_response = ImageResultLLM(
    #     name=ex_store_data["name"],
    #     first_score=ex_store_data["inn_score"],
    #     first_reason=ex_store_data["inn_reason"],
    #     first_reason_captions=ex_store_data["inn_reason_caps"],
    #     second_score=ex_store_data["seat_score"],
    #     second_reason=ex_store_data["seat_reason"],
    #     second_reason_captions=ex_store_data["seat_reason_caps"]
    # )
    
    # 프롬프트 정의
    with open("./scoring/prompts/image_1.txt", "r", encoding="utf-8") as f:
        template = f.read()
    captioner_user_prompt = template
    
    with open("./scoring/prompts/image_2.txt", "r", encoding="utf-8") as f:
        template = f.read()
    scorer_user_prompt = PromptTemplate.from_template(
        template=template,
        partial_variables={
            "ex_name": ex_name,
            "ex_captions": ex_captions,
            "ex_response": ex_response.model_dump(),
            "name": state.raw_store_data["name"]
        }
    )
    
    formatter_user_prompt = """
    ## Instruction
    - 이전의 응답을 주어진 형식에 맞게 변환합니다.
    """
    
    return {
        "captioner_user_prompt": captioner_user_prompt, 
        "scorer_user_prompt": scorer_user_prompt,
        "formatter_user_prompt": formatter_user_prompt,
        "image_contents": image_contents,
        "branch": "success"
    }

async def captioner_node(state: ImgGraphState) -> Dict[str, Any]:
    # 첫번째 LLM: 각 이미지 캡션 작성
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        thinking_budget=1024,
        timeout=60,
        max_retries=3
    )
    
    # 한번에 최대 5개 LLM 호출로 제한
    sema = asyncio.Semaphore(5)
    
    async def caption_one_image(state: ImgGraphState, image_content: Dict[str, str]):
        async with sema:
            try:
                response = await llm.ainvoke([
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": state.captioner_user_prompt},
                            image_content
                        ]
                    }
                ])
                return response
            except Exception as e:
                return e
        
    tasks = [caption_one_image(state, ic) for ic in state.image_contents]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    image_captions = ["없음" if isinstance(r, Exception) else r.content for r in responses]
            
    return {"image_captions": image_captions}

# 점수를 부여하는 LLM을 실행시키는 노드
async def scorer_node(state: ImgGraphState) -> Dict[str, Any]:
    # 3회 이상 재시도시 workflow 종료
    if state.attempts > 3:
        print(f"[내외부 점수] 재시도 횟수 초과로 스코어링 불가")
        return {"image_result": None, "branch": "too_many_attempts"}
    
    # 두번째 LLM: 내부 인테리어 평가 및 좌석 수 추정
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        thinking_budget=1024,
        timeout=60,
        max_retries=3
    )
    
    # 캡션 풀어쓰기
    captions = ""
    for index, caption in enumerate(state.image_captions):
        captions += f"{index}: {caption}\n"
    
    user_message = {
        "role": "user", 
        "content": state.scorer_user_prompt.format(captions=captions)
    }
    
    response = await llm.ainvoke(
        state.messages + [user_message], 
        tools=[GenAITool(google_search={})]
    )
    
    return {"messages": [user_message, response], "branch": "success"}

# 형식을 맞춰주는 LLM을 실행시키는 노드
async def formatter_node(state: ImgGraphState) -> Dict[str, Any]:
    # 세번째 LLM: 내부 점수 스코어링 결과를 formatting
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        thinking_budget=0,
        timeout=60,
        max_retries=3
    ).with_structured_output(ImageResultLLM)
    
    user_message = {"role": "user", "content": state.formatter_user_prompt}
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
    
    if response.first_score not in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        branch = "invalid"
        retry_reason += "1차 점수는 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 중의 하나여야 합니다.\n"
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
    for index in response.first_reason_captions:
        if index < len(image_contents):
            first_image_urls.append(image_contents[index]["url"])
    second_image_urls = []
    for index in response.second_reason_captions:
        if index < len(image_contents):
            second_image_urls.append(image_contents[index]["url"])
    
    # 1차 점수 미부여시 2차 검수 미진행으로 처리
    if response.first_score == 0:
        response.second_score = 0
        response.second_reason = "1차 점수 미부여로 2차 검수를 진행하지 않습니다."
        second_image_urls = []
    
    # 1차 점수와 2차 점수 합산으로 최종 점수 계산
    total_score = response.first_score + response.second_score
    
    image_result = ImageResult(
        name=response.name, 
        first_score=response.first_score,
        first_reason=response.first_reason,
        first_reason_captions=response.first_reason_captions,
        first_reason_images=first_image_urls,
        second_score=response.second_score,
        second_reason=response.second_reason,
        second_reason_captions=response.second_reason_captions,
        second_reason_images=second_image_urls,
        score=total_score
    )
    
    return {"image_result": image_result}