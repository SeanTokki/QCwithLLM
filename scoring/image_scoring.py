from langchain_google_genai import ChatGoogleGenerativeAI
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
from langchain.schema.runnable import RunnableLambda
import json
    
from scoring.schema import *

async def get_image_result(store_data: Dict[str, Any])-> Optional[ImageResult]:
    # 첫번째 LLM에 넘길 이미지 contents 생성
    if store_data["image_list"] or store_data["inner_image_list"]:
        image_list = store_data["image_list"] + store_data["inner_image_list"]
    else:
        print(f"[매장 내부 점수] 이미지가 없어서 스코어링 불가")
        return None
    image_content = []
    for image_url in image_list:
        image_content.append({"type": "image", "source_type": "url", "url": image_url})
        
    # example용 데이터
    with open("./data/examples/example_scored_store_data.json", "r", encoding="utf-8") as f:
        ex_store_data = json.load(f)
    ex_response = ImageResultLLM(
        name=ex_store_data["name"],
        first_score=ex_store_data["inn_score"],
        first_reason=ex_store_data["inn_reason"],
        first_reason_image=1,
        second_score=ex_store_data["seat_score"],
        second_reason=ex_store_data["seat_reason"],
        second_reason_image=2
    )

    # 첫번째 LLM에게 넘길 프롬프트
    system_prompt = """
    ## Role
    당신은 매장 분석 전문가입니다.
    당신은 특정 매장이 제휴를 맺기 적합한 매장인지 판단하기 위해 점수 기준표에 따라 매장의 이미지를 보고 점수를 부여하는 역할을 맡고 있습니다.
    차근차근 단계별로 생각하며 지시를 수행하세요.
    """
    user_prompt = """
    ## Instruction
    - "Scoring Rule"을 참고하여 입력으로 주어진 매장의 이미지를 보고 스코어링을 진행합니다.
    - 주어진 매장 이미지의 인덱스는 1부터 시작합니다.
    - 1차 검수를 진행하여 점수를 부여하고 그 근거를 서술합니다.
    - 주어진 이미지 중 1차 검수 과정에서 가장 영향을 크게 미친 하나를 골라 근거로써 첨부할 수 있습니다.
    - 1차 검수에서 주어진 이미지만으로 판단이 불가능할 경우 0점을 부여하고 그 이유를 서술합니다.
    - 2차 검수를 진행하여 점수를 부여하고 그 근거를 서술합니다.
    - 주어진 이미지 중 2차 검수 과정에서 가장 영향을 크게 미친 하나를 골라 근거로써 첨부할 수 있습니다.
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
    user_prompt = user_prompt.format(
        ex_name=ex_store_data["name"],
        ex_response=ex_response.model_dump(),
        name=store_data["name"]
        )
    
    # 첫번째 LLM: 이미지 스코어링 진행
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
                    "content": [{"type": "text", "text": args["user_prompt"]}] + args["image_content"]
                }
            ],
            tools=[GenAITool(google_search={})],
        )
        
        return text_response.content

    # 하나의 chain으로 묶기 위해 함수로 감싸기
    first_llm = RunnableLambda(first_llm_invoke)

    # 두번째 LLM: 이미지 스코어링 결과를 formatting
    structured_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        thinking_budget=0,
    ).with_structured_output(ImageResultLLM)

    chain = first_llm | structured_llm
    
    try:
        response: ImageResultLLM = await chain.ainvoke({"system_prompt": system_prompt, "user_prompt": user_prompt, "image_content": image_content})
    except Exception as e:
        print(f"[매장 내부 점수] 오류 발생: {e}")
        return None
    
    if response:
        # 인덱스로 주어진 image 정보를 url string으로 변환
        first_image_url = image_content[response.first_reason_image-1]["url"] if response.first_reason_image else None
        second_image_url = image_content[response.second_reason_image-1]["url"] if response.second_reason_image else None
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
            first_reason_image=first_image_url,
            second_score=response.second_score,
            second_reason=response.second_reason,
            second_reason_image=second_image_url,
            score=total_score
        )
    else:
        image_result = None
    
    return image_result