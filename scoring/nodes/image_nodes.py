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
    0: 전반적으로 어두운 톤의 인테리어로, 붉은색 가죽 소파와 어두운 우드 톤의 테이블이 배치되어 있다. 벽면에는 직사각형 형태의 조명이 설치되어 있으며, 천장에는 아치형 구조물에 패턴이 새겨져 있다.
    1: 붉은 벽돌 기둥과 아치형 구조물이 특징인 레스토랑이다. 짙은 와인색 가죽 소파와 흰색 테이블보가 깔린 테이블이 줄지어 배치되어 있다. 전반적으로 고풍스럽고 차분한 분위기를 연출한다.
    2: 붉은색 벽과 어두운 갈색의 가죽 소파가 조화를 이루는 실내다. 천장은 아치형으로 디자인되었으며, 금색과 검은색의 모자이크 패턴으로 장식되어 있다. 전반적으로 고풍스러운 분위기를 연출한다.
    3: 전체적으로 어두운 우드 톤으로 마감된 전형적인 매장이다. 카운터 안쪽으로 주방 공간이 보이며, 특별한 테마나 컨셉은 보이지 않는다.
    4: 전체적으로 붉은색 계열의 벽과 어두운 갈색의 의자, 테이블이 조화를 이루는 레스토랑이다. 벽면은 붉은색의 물결무늬 패널과 벽돌 아치형 구조물로 이루어져 있으며, 천장은 노란색 계열의 마감재와 간접 조명으로 따뜻한 분위기를 연출한다.
    5: 카운터는 어두운 우드 톤으로 되어 있으며, 벽에는 소 그림 액자가 걸려 있다. 천장은 금색과 검은색 체크무늬로 장식되어 있어 고급스러운 분위기를 연출한다.
    6: 어두운 톤의 목재와 붉은색 가죽 의자가 조화를 이루는 레스토랑이다. 천장은 노란색으로 마감되었고, 벽면에는 붉은색 물결무늬 장식이 있어 이국적인 분위기를 연출한다.
    7: 붉은 벽돌과 붉은색 소파가 조화를 이루는 레스토랑이다. 천장은 금색과 모자이크 타일로 장식되어 있으며, 아치형 통로와 벽등이 고풍스러운 분위기를 더한다.
    8: 붉은 벽돌과 아치형 구조물, 그리고 천장의 문양 장식이 고풍스러운 분위기를 연출한다. 짙은 갈색의 가죽 소파와 의자가 배치되어 있으며, 테이블은 흰색 상판으로 되어 있다.
    9: 붉은 벽돌과 붉은색 소파, 그리고 붉은색 벽이 조화를 이루는 매장이다. 천장은 노란색과 모자이크 타일로 장식되어 있으며, 아치형 통로와 벽등이 고풍스러운 분위기를 더한다.
    """
    inn_ex_response = f"""
    name: {ex_name}
    inn_score: 4.0
    inn_reason: 캡션 0, 1, 2, 4, 5, 6, 7, 8, 9에서 붉은 벽돌, 아치형 구조물, 붉은색 가죽 소파, 금색/모자이크 천장 등 고풍스럽고 이국적인 분위기가 일관되게 언급되어 특정 컨셉(유럽풍 또는 고전적인 분위기)이 매장 전체에 적용된 것으로 판단됨. 캡션 3에서 '특별한 테마나 컨셉은 보이지 않는다'고 언급되었으나, 다른 다수의 캡션에서 명확한 컨셉 요소가 반복적으로 나타나므로 4점으로 상향 조정함.
    inn_reason_idxs: [0, 1, 2, 4, 5, 6, 7, 8, 9]
    """
    seat_ex_response = f"""
    name: {ex_name}
    seat_score: 0.5
    seat_reason: 1, 2, 5, 7, 8, 9에서 확인된 좌석 합계가 32석으로 30석 초과 구간에 해당하여 조정점수 +0.5.
    seat_reason_idxs: [1, 2, 5, 7, 8, 9]
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
    with open("./scoring/prompts/image_captioner.txt", "r", encoding="utf-8") as f:
        template = f.read()
    captioner_user_prompt = template
    
    with open("./scoring/prompts/image_inn_scorer.txt", "r", encoding="utf-8") as f:
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
    
    with open("./scoring/prompts/image_seat_scorer.txt", "r", encoding="utf-8") as f:
        template = f.read()
    seat_scorer_user_prompt = template.format(
        ex_name=ex_name,
        ex_response=seat_ex_response,
        name=state.raw_store_data["name"]
    )
    
    with open("./scoring/prompts/image_formatter.txt", "r", encoding="utf-8") as f:
        template = f.read()
    formatter_user_prompt = PromptTemplate.from_template(
        template=template
    )
    
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
        timeout=30,
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
        
    tasks = [caption_one_image(state, ic) for ic in state.image_contents[:10]]
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
        timeout=30,
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
    
    return {"messages": [user_message, response], "inn_response": response.content}

# 좌석수 점수를 부여하는 LLM을 실행시키는 노드
async def seat_scorer_node(state: ImgGraphState) -> Dict[str, Any]:
    # 세번째 LLM: 좌석수 추정
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        thinking_budget=0,
        timeout=30,
        max_retries=3
    )
    
    user_message = {
        "role": "user", 
        "content": [state.seat_scorer_user_prompt] + state.image_contents[:10]
    }
    
    response = await llm.ainvoke(
        state.messages + [user_message], 
        tools=[GenAITool(google_search={})]
    )
    
    return {"messages": [user_message, response], "seat_response": response.content}

# 형식을 맞춰주는 LLM을 실행시키는 노드
async def formatter_node(state: ImgGraphState) -> Dict[str, Any]:
    # 세번째 LLM: 내부 점수 스코어링 결과를 formatting
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        thinking_budget=0,
        timeout=30,
        max_retries=3
    ).with_structured_output(ImageResultLLM)
    
    user_message = {
        "role": "user", 
        "content": state.formatter_user_prompt.format(
            inn_response=state.inn_response,
            seat_response=state.seat_response
        )
    }
    response = await llm.ainvoke([user_message])
    
    return {"image_result": response}

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