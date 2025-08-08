from typing import List, Dict, Any, Optional, Annotated, Union
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langgraph.graph.state import CompiledStateGraph
from langchain.prompts import PromptTemplate

class ImageResultLLM(BaseModel):
    """매장 이미지 평가 결과(LLM 출력)"""
    name: str = Field(description="매장 이름 (입력과 동일)")
    inn_score: float = Field(description="내부 점수")
    inn_reason: str = Field(description="내부 점수 부여에 대한 근거")
    inn_reason_idxs: List[int] = Field(description="내부 점수 부여에 영향을 미친 이미지 인덱스들")
    seat_score: float = Field(description="좌석수 점수")
    seat_reason: str = Field(description="좌석수 점수 부여에 대한 근거")
    seat_reason_idxs: List[int] = Field(description="좌석수 점수 부여에 영향을 미친 이미지 인덱스들")

class ImageResult(ImageResultLLM):
    """매장 이미지 평가 결과"""
    inn_reason_images: List[str] = Field(description="내부 점수 부여에 영향을 미친 이미지 urls")
    seat_reason_images: List[str] = Field(description="좌석수 점수 부여에 영향을 미친 이미지 urls")
    score: float = Field(description="내부 점수와 좌석수 점수의 합산")

class ImgGraphState(BaseModel):    
    """매장 이미지 평가 그래프 상태"""
    raw_store_data: Dict[str, Any]
    messages: Annotated[list, add_messages]
    image_contents: List[Dict[str, str]] = Field(default_factory=list)
    inn_response: Optional[str | list[str | dict]] = None
    seat_response: Optional[str | list[str | dict]] = None
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
    """전체 스코어링 결과"""
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
    inn_reason_idxs: List[int]
    inn_reason_imgs: List[str]
    seat_score: float
    seat_reason: str
    seat_reason_idxs: List[int]
    seat_reason_imgs: List[str]
    img_score: float
    add_items: List[AdditionalItem]
    add_score: float
    tot_score: float
    
class QueryTaskPayload(BaseModel):
    """POST /tasks 요청의 payload"""
    query: str

class NaverIDPayload(BaseModel):
    """POST /internal/callback/nid-found 요청의 payload"""
    query: str
    naver_id: Optional[str]
    need_crawling: bool
    
class CrawlCompPayload(BaseModel):
    """POST /internal/callback/crawl-completed 요청의 payload"""
    naver_id: str
    success: bool