from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
import json
import asyncio

from schema import *

# 전처리 노드
def preprocessor_node(state: ImgGraphState) -> Dict[str, Any]:
    store_data = state.raw_store_data
    
    # LLM에 넘길 내부 이미지 contents 생성
    if store_data["inner_image_list"]:
        image_list = store_data["inner_image_list"]
    else:
        print(f"[내외부 점수] 이미지가 없어서 스코어링 불가")
        return {"image_result": None, "branch": "no_image"}
    image_contents = []
    for image_url in image_list:
        image_contents.append({"type": "image", "source_type": "url", "url": image_url})
    
    return {
        "image_contents": image_contents,
        "branch": "success"
    }

# 이미지 캡션 생성 노드
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
    
    # 캡셔닝 지시문
    with open("./scoring/prompts/image_captioner.txt", "r", encoding="utf-8") as f:
        template = f.read()
    instruction = template
    
    # 한번에 최대 5개 LLM 호출로 제한
    sema = asyncio.Semaphore(5)
    
    async def caption_one_image(state: ImgGraphState, image_content: Dict[str, str]):
        async with sema:
            try:
                response = await llm.ainvoke([
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
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

# 캡션으로 내부 점수를 부여하는 LLM을 실행시키는 노드
async def inn_scorer_with_caps_node(state: ImgGraphState) -> Dict[str, Any]:
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
    
    # 스코어링 지시문
    with open("./scoring/prompts/image_inn_scorer_with_caps.txt", "r", encoding="utf-8") as f:
        template = f.read()
    instruction = template + "\n\n"
    
    # few-shot 프롬프트
    with open("./scoring/prompts/image_inn_caption_few_shot.txt", "r", encoding="utf-8") as f:
        template = f.read()
    instruction += template.format(name=state.raw_store_data.get('name'), captions=captions)
    
    user_message = {
        "role": "user", 
        "content": instruction
    }
    
    response = await llm.ainvoke(
        state.messages + [user_message]
    )
    
    return {"messages": [user_message, response], "inn_response": response.content}

# 두 노드를 동시에 실행시키기 위해 필요한 더미 노드
def dispatcher_node(state: ImgGraphState) -> Dict[str, Any]:
    
    return{}

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
    
    # LLM에 최종적으로 들어갈 input
    input_messages = []
    
    # 스코어링 지시 message
    with open("./scoring/prompts/image_inn_scorer.txt", "r", encoding="utf-8") as f:
        template = f.read()
    instruction_message = {
        "role": "user",
        "content": template
    }
    
    # few-shot messages
    with open("./data/examples/inn_few_shot_messages.json", "r", encoding="utf-8") as f:
        inn_few_shot_messages: list = json.load(f)
    
    # 진짜로 묻고싶은 input message    
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"'{state.raw_store_data.get('name')}' 매장에 대한 평가를 해주세요."
            }
        ] + state.image_contents[:10]
    }
    
    input_messages += state.messages
    input_messages.append(instruction_message)
    input_messages += inn_few_shot_messages # state.messages 기록에는 들어가지 않음
    input_messages.append(user_message)
    
    response = await llm.ainvoke(input_messages)
    
    return {"messages": [instruction_message, user_message, response], "inn_response": response.content}

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
    
    # example용 가상 매장 데이터
    ex_name = "쟈니 다이너"
    seat_ex_response = f"""name: {ex_name}
    seat_score: 0.5
    seat_reason: 1, 2, 5, 7, 8, 9에서 확인된 좌석 합계가 32석으로 30석 초과 구간에 해당하여 조정점수 +0.5.
    seat_reason_idxs: [1, 2, 5, 7, 8, 9]
    """
    
    # 스코어링 지시문
    with open("./scoring/prompts/image_seat_scorer.txt", "r", encoding="utf-8") as f:
        template = f.read()
    instruction = template.format(
        ex_name=ex_name,
        ex_response=seat_ex_response,
        name=state.raw_store_data["name"]
    )
    
    user_message = {
        "role": "user", 
        "content": [
            {
                "type": "text",
                "text": instruction
            }
        ] + state.image_contents[:10]
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
    
    # formatting 지시문
    with open("./scoring/prompts/image_formatter.txt", "r", encoding="utf-8") as f:
        template = f.read()
    instruction = PromptTemplate.from_template(
        template=template
    )
    
    user_message = {
        "role": "user", 
        "content": instruction.format(
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
    
    retry_reason = "평가 결과가 잘못되었으므로 재평가하세요.\n잘못된 이유:\n"
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
            if state.image_captions:
                inn_image_urls.append(image_contents[index]["url"])
            else:
                inn_image_urls.append(image_contents[index-1]["url"])
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