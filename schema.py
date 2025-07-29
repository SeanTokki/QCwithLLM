from typing import List, Dict, Any, Optional, Annotated, Union
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langgraph.graph.state import CompiledStateGraph

class ImageResultLLM(BaseModel):
    """매장 이미지 평가 결과(LLM 출력)"""
    name: str = Field(description="매장 이름 (입력과 동일)")
    first_score: float = Field(description="1차 점수")
    first_reason: str = Field(description="1차 점수 부여에 대한 근거")
    first_reason_image: Optional[int] = Field(description="1차 점수 부여에 가장 큰 영향을 미친 이미지 인덱스(1부터 시작)")
    second_score: float = Field(description="2차 점수")
    second_reason: str = Field(description="2차 점수 부여에 대한 근거")
    second_reason_image: Optional[int] = Field(description="2차 점수 부여에 가장 큰 영향을 미친 이미지 인덱스(1부터 시작)")

class ImageResult(ImageResultLLM):
    """매장 이미지 평가 결과"""
    first_reason_image: Optional[str] = Field(description="1차 점수 부여에 가장 큰 영향을 미친 이미지 url")
    second_reason_image: Optional[str] = Field(description="2차 점수 부여에 가장 큰 영향을 미친 이미지 url")
    score: float = Field(description="1차 점수와 2차 점수의 합산")

class ImgGraphState(BaseModel):    
    """매장 이미지 평가 그래프 상태"""
    raw_store_data: Dict[str, Any]
    messages: Annotated[list, add_messages]
    first_user_prompt: Optional[str] = None
    second_user_prompt: Optional[str] = None
    image_contents: List[Dict[str, str]] = Field(default_factory=list)
    image_result: Union[ImageResult, ImageResultLLM, None] = None
    branch: Optional[str] = None
    attempts: int = 1

class CategoryResult(BaseModel):
    """카테고리 매칭 결과"""
    name: str = Field(description="매장 이름 (입력과 동일)")
    top_category: str = Field(description="매칭된 상위 카테고리")
    sub_category: str = Field(description="매칭된 하위 카테고리")
    score: int= Field(description="매칭된 하위 카테고리의 점수")
    reason: str = Field(description="카테고리 매칭의 판단 근거")
    
class CatGraphState(BaseModel):    
    """카테고리 매칭 그래프 상태"""
    raw_store_data: Dict[str, Any]
    messages: Annotated[list, add_messages]
    first_user_prompt: Optional[str] = None
    second_user_prompt: Optional[str] = None
    matching_result: Optional[CategoryResult] = None
    branch: Optional[str] = None
    attempts: int = 1
    feedback: Optional[str] = None
    ok: bool = False

class AdditionalItem(BaseModel):
    """하나의 추가 점수 항목"""
    item: str = Field(description="추가 점수 항목")
    score: float = Field(description="이 항목에 대한 점수")
    selected: bool = Field(description="이 항목이 해당하는지 여부")
    reason: str = Field(description="항목 해당 여부 판단 근거")

class AdditionalResultLLM(BaseModel):
    """추가 점수 항목 평가 결과(LLM 출력)"""
    name: str = Field(description="매장 이름 (입력과 동일)")
    items: List[AdditionalItem] = Field(description="이 매장에 해당하는 추가 점수 항목의 리스트")
    
class AdditionalResult(AdditionalResultLLM):
    """추가 점수 항목 평가 결과"""
    score: float = Field(description="해당하는 모든 항목에 대한 추가 점수의 합산 결과")

class AddGraphState(BaseModel):    
    """추가 점수 부여 그래프 상태"""
    raw_store_data: Dict[str, Any]
    messages: Annotated[list, add_messages]
    first_user_prompt: Optional[str] = None
    second_user_prompt: Optional[str] = None
    additional_result: Union[AdditionalResult, AdditionalResultLLM, None] = None
    branch: Optional[str] = None
    attempts: int = 1

class PositionResult(BaseModel):
    """위치 점수"""
    name: str = Field(description="매장 이름 (입력과 동일)")
    score: float = Field(description="위치 점수")
    reason: str = Field(description="위치 점수 선정의 이유")

class FullResult(BaseModel):
    naver_id: str
    name: str
    pos_score: float
    pos_reason: str
    top_cat: str
    sub_cat: str
    cat_reason: str
    cat_score: float
    inn_score: float
    inn_reason: str
    inn_reason_img: Optional[str]
    seat_score: float
    seat_reason: str
    seat_reason_img: Optional[str]
    img_score: float
    add_items: List[AdditionalItem]
    add_score: float
    tot_score: float