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
    inn_ex_response = f"""
    name: {ex_name}
    inn_score: 3.0
    inn_reason: 캡션 0-4 모두에서 특별한 테마나 독창적 소품 없이 우드·화이트 톤의 일반적인 카페 인테리어만 확인됨.
    inn_reason_idxs: [0, 1, 2, 3, 4]
    """
    seat_ex_response = f"""
    name: {ex_name}
    seat_score: 0.0
    seat_reason: 이미지 1·4에서 확인된 좌석 합계가 26석으로 11-30석 구간에 해당하여 조정점수 0.
    seat_reason_idxs: [1, 4]
    """

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
    inn_scorer_user_prompt = PromptTemplate.from_template(
        template=template,
        partial_variables={
            "ex_name": ex_name,
            "ex_captions": ex_captions,
            "ex_response": inn_ex_response,
            "name": state.raw_store_data["name"]
        }
    )
    
    with open("./scoring/prompts/image_3.txt", "r", encoding="utf-8") as f:
        template = f.read()
    seat_scorer_user_prompt = template.format(
        ex_name=ex_name,
        ex_response=seat_ex_response,
        name=state.raw_store_data["name"]
    )
    
    formatter_user_prompt = """
    ## Instruction
    - 이전의 응답을 주어진 형식에 맞게 변환합니다.
    """
    
    return {
        "captioner_user_prompt": captioner_user_prompt, 
        "inn_scorer_user_prompt": inn_scorer_user_prompt,
        "seat_scorer_user_prompt": seat_scorer_user_prompt,
        "formatter_user_prompt": formatter_user_prompt,
        "image_contents": image_contents,
        "branch": "success"
    }

async def captioner_node(state: ImgGraphState) -> Dict[str, Any]:
    # 이미 캡션이 있으면 다음 노드로 이동
    if state.image_captions:
        return {}
    
    # 첫번째 LLM: 각 이미지 캡션 작성
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        thinking_budget=0,
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

# 내부 점수를 부여하는 LLM을 실행시키는 노드
async def inn_scorer_node(state: ImgGraphState) -> Dict[str, Any]:
    # 두번째 LLM: 내부 인테리어 평가
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        thinking_budget=0,
        timeout=60,
        max_retries=3
    )
    
    # 캡션 풀어쓰기
    captions = ""
    for index, caption in enumerate(state.image_captions):
        captions += f"{index}: {caption}\n"
    
    user_message = {
        "role": "user", 
        "content": state.inn_scorer_user_prompt.format(captions=captions)
    }
    
    response = await llm.ainvoke(
        state.messages + [user_message]
    )
    
    return {"messages": [user_message, response]}

# 좌석수 점수를 부여하는 LLM을 실행시키는 노드
async def seat_scorer_node(state: ImgGraphState) -> Dict[str, Any]:
    # 세번째 LLM: 좌석수 추정
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        thinking_budget=0,
        timeout=60,
        max_retries=3
    )
    
    user_message = {
        "role": "user", 
        "content": [state.seat_scorer_user_prompt] + state.image_contents
    }
    
    response = await llm.ainvoke(
        state.messages + [user_message], 
        tools=[GenAITool(google_search={})]
    )
    
    return {"messages": [user_message, response]}

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
    
    need_inn = response.inn_score not in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    need_seat = response.seat_score not in [0.0, 0.5, -0.5]
    
    # 이상이 없다면 다음 노드로 진행
    if not (need_inn or need_seat):
        return {"branch": "valid"}
    
    # 3회 이상 재시도시 workflow 종료
    if state.attempts + 1 > 3:
        print(f"[내외부 점수] 재시도 횟수 초과로 스코어링 불가")
        return {
            "image_result": None,
            "branch": "too many attempts",
            "attempts": state.attempts + 1,
        }
    
    retry_reason = """
    평가 결과가 잘못되었으므로 재평가하세요.
    잘못된 이유: 
    """
    if need_inn:
        retry_reason += "내부 점수는 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 중의 하나여야 합니다.\n"
    if need_seat:
        retry_reason += "좌석수 점수는 0.0, 0.5, -0.5 중의 하나여야 합니다.\n"

    # 이상이 있다면 틀린 부분을 알려주고 재시도
    return {
        "branch": "invalid both" if (need_inn and need_seat) 
                else "invalid inn" if (need_inn) else "invalid seat",
        "attempts": state.attempts + 1,
        "messages": [{"role": "user", "content": retry_reason}]
    }

# LLM 결과에 이어서 추가적인 가공을 하는 노드
def postprocessor_node(state: ImgGraphState) -> Dict[str, Any]:
    response = state.image_result
    image_contents = state.image_contents
    
    # 인덱스로 주어진 image 정보를 url string으로 변환
    inn_image_urls = []
    for index in response.inn_reason_idxs:
        if index < len(image_contents):
            inn_image_urls.append(image_contents[index]["url"])
    seat_image_urls = []
    for index in response.seat_reason_idxs:
        if index <= len(image_contents):
            seat_image_urls.append(image_contents[index-1]["url"])
    
    # 내부 점수 미부여시 좌석수 검수 진행 안함
    if response.inn_score == 0:
        response.seat_score = 0
        response.seat_reason = "내부 점수 미부여로 좌석수 검수를 진행하지 않습니다."
        seat_image_urls = []
    
    # 내부 점수와 좌석수 점수 합산으로 최종 점수 계산
    total_score = response.inn_score + response.seat_score
    
    image_result = ImageResult(
        name=response.name, 
        inn_score=response.inn_score,
        inn_reason=response.inn_reason,
        inn_reason_idxs=response.inn_reason_idxs,
        inn_reason_images=inn_image_urls,
        seat_score=response.seat_score,
        seat_reason=response.seat_reason,
        seat_reason_idxs=response.seat_reason_idxs,
        seat_reason_images=seat_image_urls,
        score=total_score
    )
    
    return {"image_result": image_result}