# LLM 활용 매장 QC 점수 자동화 시스템

## 1. 프로젝트 개요

이 프로젝트는 네이버 지도에서 수집된 매장 데이터를 기반으로, LLM을 활용하여 QC 점수를 자동으로 부여하는 시스템입니다.

FastAPI로 구축된 비동기 웹 서버를 통해 naver ID 기반으로 매장 점수 조회를 요청하고, 만약 미리 계산된 점수가 없다면 스코어링 작업을 시작합니다. 이 때, 복잡하고 시간이 소요되는 LLM 기반 스코어링 작업은 백그라운드에서 처리됩니다. 이를 통해 사용자는 작업이 완료될 때까지 기다릴 필요 없이 즉시 작업 ID를 발급받고, 나중에 해당 ID로 진행 상태나 결과를 조회할 수 있습니다.

또한, 외부 크롤러와 이벤트 기반으로 연동하여, 특정 검색어(예: "강남역 맛집")만으로도 매장 검색, 데이터 수집, 점수 계산까지 이어지는 전체 워크플로우를 자동화합니다.

---

## 2. 핵심 기술 스택

- **Backend Framework**: `FastAPI`
- **LLM Orchestration**: `LangChain`, `LangGraph`
- **LLM Provider**: `Google Gemini (google-genai)`
- **Vector Database**: `ChromaDB` (유사도 기반 사례 검색)
- **Text Embedding**: `Sentence-Transformers`
- **Geospatial Analysis**: `Shapely`
- **AWS SDK**: `boto3` (AWS 서비스 연동을 통한 이벤트 메시징)

---

## 3. 주요 구성 요소 및 아키텍처

1.  **FastAPI Application (`app.py`)**:
    - 클라이언트 요청을 수신하여 작업을 수행합니다.
    - 점수 조회, 작업 상태 확인 등 핵심 API 엔드포인트를 제공합니다.
    - 시간이 오래 걸리는 스코어링 작업은 `BackgroundTasks`를 통해 즉시 백그라운드로 위임하고, 클라이언트에게는 `task_id`를 반환합니다.
    - 애플리케이션 시작 시, `LangGraph`로 정의된 스코어링 그래프(Category, Image, Additional)를 미리 빌드하여 메모리에 상주시켜, 요청 처리 시의 오버헤드를 최소화합니다.

2.  **Scoring Pipeline (`scoring/` 폴더)**:
    - ⚠️ **현재는 로컬 DB(json 파일)를 활용하도록 구현되어 있습니다.**
    - 백그라운드에서 실행되는 핵심 로직입니다.
    - 위치 점수 계산, LLM 기반 점수(메뉴, 내외부, 추가) 계산 등 전체 스코어링 과정을 총괄합니다.
    - `LangGraph`로 구성된 각 점수별 에이전트를 병렬적으로 실행하고, 최종 결과를 취합하여 데이터베이스에 저장합니다.

3.  **Vector Database (`utils/build_store_vector_db.py`)**:
    - `ChromaDB`를 사용하여 매장 정보의 벡터 저장소를 구축합니다.
    - 각 매장의 텍스트 정보(카테고리, 메뉴, 리뷰 등)를 `SentenceTransformer`로 임베딩하여 벡터로 저장합니다.
    - **(용도)** '메뉴 점수' 평가 시, 현재 평가할 매장과 유사한 매장의 과거 평가 사례를 검색하는 `similar_partner_search` 도구의 기반 기술로 사용됩니다. 이를 통해 LLM이 더 일관성 있는 판단을 내리도록 돕는 **보조 기억 장치** 역할을 합니다.

4.  **Batch-Scoring & Evaluation (`main.py`, `evaluation.py`)**:
    - `main.py`: FastAPI 서버와 별개로, 로컬에서 특정 매장 목록에 대한 스코어링을 일괄 실행하기 위한 스크립트입니다. 테스트 데이터 생성이나 대규모 재평가 시 사용됩니다.
    - `evaluation.py`: 스코어링 모델의 성능을 정량적으로 평가하는 스크립트입니다. Ground Truth 데이터와 모델 예측 결과를 비교하여 `MAE`, `Quadratic Cohen's Kappa`, `Accuracy` 세 가지 성능 지표를 계산하고, 비교 테이블을 생성합니다.

5.  **External Crawler (외부 시스템)**:
    - ⚠️ **아직 외부 크롤러와 이벤트 송수신 테스트는 진행하지 못했습니다.**
    - API 서버와 분리된 독립적인 크롤링 시스템입니다.
    - API 서버로부터 `EventBridge`(`utils/event.py`)를 통해 특정 `query`에 대한 크롤링 요청 이벤트를 수신합니다.
    - 네이버 지도에서 해당 `query`에 맞는 매장을 찾아 `naver_id`를 획득하고, 필요한 상세 데이터(리뷰, 이미지 등)를 크롤링합니다.
    - naver ID를 찾았을 때와 데이터 수집이 완료되었을 때 API 서버의 `/internal/callback/*` 엔드포인트를 호출하여 후속 작업을 트리거합니다.

---

## 4. 점수 계산 로직
모든 평가 기준 및 프롬프트는 QC Center 스코어링 기준 표를 참고하여 작성하였습니다.

**총점 = (위치 점수 + 메뉴 점수 + 내외부 점수) / 3 + 추가 점수**

-   **위치 점수 (`scoring/position_scoring.py`)**:
    - LLM을 사용하지 않는 **알고리즘 기반** 점수입니다.
    - `pandas`와 `shapely`를 사용하여, 매장의 GPS 좌표가 미리 정의된 "핫플레이스" 폴리곤(`data/polygons/`) 내에 위치하는지 여부와 지하철역과의 거리를 계산하여 점수를 부여합니다.

-   **메뉴 점수 (`scoring/graphs/category_graph.py`, `scoring/nodes/category_nodes.py`)**:
    - **LLM 기반** 점수입니다.
    - 매장의 이름, 카테고리, 메뉴, 리뷰 데이터를 분석하여, `data/rules/category_score_rule.json`에 정의된 기준표에 가장 적합한 상위/하위 카테고리를 매칭하고 점수를 부여합니다.
    - 에이전트가 사용할 수 있는 도구로 유사 매장의 카테고리 매칭 정보를 불러올 수 있는 `similar_partner_search` 도구와, 해당 매장의 음식/음료 이미지를 불러올 수 있는 `get_food_images` 도구가 구현되어 있습니다.
    - 프롬프트는 `scoring/prompts/category_*.txt` 파일에 정의되어 있습니다.

-   **내외부 점수 (`scoring/graphs/image_graph.py`, `scoring/nodes/image_nodes.py`)**:
    - **LLM(멀티모달) 기반** 점수입니다.
    - 수집된 매장 이미지를 분석하여 인테리어 컨셉, 분위기, 좌석 수 등을 평가하고 점수를 부여합니다.
    - 판단 근거가 된 이미지를 결과에 포함시킬 수 있습니다.
    - 에이전트는 좌석 수 추정에 `Google Search Gemini Built-in tool`을 활용할 수 있습니다.
    - 프롬프트는 `scoring/prompts/image_*.txt` 파일에 정의되어 있습니다.

-   **추가 점수 (`scoring/graphs/additional_graph.py`, `scoring/nodes/additional_nodes.py`)**:
    - **LLM 기반** 점수이며, 필요시 `Google Search Gemini Built-in Tool`을 활용합니다.
    - TV 프로그램 방영 여부, 유명인 운영, 예약 정책 등 수집된 데이터만으로는 알기 어려운 정보를 찾고 기준에 맞게 점수를 부여합니다.
    - 프롬프트는 `scoring/prompts/additional_*.txt` 파일에 정의되어 있습니다.

---

## 5. 주요 실행 흐름 (API Workflows)

### 5.1. `naver_id`로 직접 점수 요청

1.  클라이언트가 특정 매장의 점수를 `GET /stores/{naver_id}/score`로 요청합니다.
2.  API 서버는 `db.py`를 통해 해당 `naver_id`의 점수가 이미 캐시(저장)되어 있는지 확인합니다.
    - **(A) 점수가 존재할 경우**: 즉시 저장된 점수를 `{"ready": true, "score": ...}` 형태로 반환합니다.
    - **(B) 점수가 존재하지 않을 경우**:
        1.  새로운 `task`를 생성하고, `scoring_task` 함수를 백그라운드에서 실행하도록 예약합니다.
        2.  클라이언트에게는 즉시 `202 Accepted` 상태 코드와 함께 `{"ready": false, "task_id": "..."}`를 반환합니다.
3.  클라이언트는 발급받은 `task_id`를 사용하여 `GET /tasks/{task_id}`를 주기적으로(polling) 호출하여 작업의 진행 상태(`status`, `progress`)를 확인합니다.
4.  작업이 완료되면(`status: "SUCCESS"`), 클라이언트는 다시 `GET /stores/{naver_id}/score`를 호출하여 최종 점수를 획득합니다.

### 5.2. `query`로 점수 요청 (크롤러 연동)

1.  클라이언트가 특정 검색어로 `POST /tasks` (`{"query": "신사역 카페"}`)를 요청합니다.
2.  API 서버는 `task`를 생성하고, `EventBridge`를 통해 외부 크롤러에게 `find_nid_and_crawl` 이벤트를 전송합니다.
3.  클라이언트에게는 `task_id`를 즉시 반환합니다.
4.  **(외부 크롤러)** 이벤트를 수신하고, "신사역 카페"에 해당하는 매장의 `naver_id`를 찾고 필요시 데이터를 크롤링합니다.
5.  크롤러는 수집 결과를 가지고 API 서버의 콜백 엔드포인트 `POST /internal/callback/nid-found` 또는 `crawl-completed`를 호출합니다.
6.  콜백 엔드포인트는 해당 `task`의 상태를 업데이트하고, **5.1 (B)** 단계와 동일하게 백그라운드에서 `scoring_task`를 실행하여 점수 계산을 시작합니다.
7.  이후 과정은 5.1과 동일합니다.

---

## 6. 프로젝트 구조

```
C:.
├── app.py              # FastAPI 애플리케이션 (라우터, API 엔드포인트)
├── main.py             # 로컬에서 스코어링을 일괄 실행하기 위한 스크립트
├── evaluation.py       # 성능 평가 스크립트
├── db.py               # 데이터베이스 CRUD 함수
├── schema.py           # API 요청/응답 모델 (Pydantic)
├── requirements.txt    # Python 패키지 의존성 목록
├── .env                # 환경 변수 설정 파일 (GOOGLE_API_KEY 등)
├── data/               # 규칙, 폴리곤, 예시 데이터 등 정적 데이터
|   ├── examples/       # LLM에게 제공될 few-shot examples
|   ├── graphs/         # 그래프를 실행시켰을 때 저장되는 그래프 시각화 이미지
│   ├── rules/          # 점수 계산에 사용되는 규칙 JSON 파일
│   ├── polygons/       # 위치 점수 계산을 위한 지역 폴리곤 CSV
│   └── test/           # 테스트용 로컬 DB(json 파일)
├── scoring/            # 핵심 스코어링 로직
│   ├── pipeline.py     # 전체 스코어링 파이프라인을 조율하는 메인 함수
│   ├── position_scoring.py # 위치 점수 계산 로직
│   ├── graphs/         # LangGraph 기반 스코어링 에이전트(그래프) 정의
│   │   ├── category_graph.py
│   │   ├── image_graph.py
│   │   └── additional_graph.py
|   ├── nodes/          # LangGraph 에이전트를 구성하는 노드 정의
│   │   ├── category_nodes.py
│   │   ├── image_nodes.py
│   │   └── additional_nodes.py
│   └── prompts/        # LLM에게 전달될 프롬프트 템플릿
└── utils/              # 유틸리티 모음
    ├── event.py        # 외부 시스템과 통신하기 위한 EventBridge
    └── build_store_vector_db.py # ChromaDB 벡터 저장소 구축 스크립트
```

---

## 7. 설치 및 실행 방법

### 7.1. 사전 준비

-   Python 3.11

### 7.2. 설치

1.  **저장소 복제**
    ```bash
    git clone <repository-url>
    cd <project-name>
    ```

2.  **가상 환경 생성 및 활성화**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **의존성 설치**
    ```bash
    pip install -r requirements.txt
    ```

4.  **환경 변수 설정**
    -   프로젝트 루트에 `.env` 파일을 생성합니다.
    -   필요한 API KEY를 작성합니다.
    ```
    GOOGLE_API_KEY=""
    AWS_EVENT_ACCESS_KEY_ID=""
    AWS_EVENT_SECRET_ACCESS_KEY=""
    ```

5.  **벡터 DB 구축**
    - ⚠️ **현재 레포지토리에 있는 벡터 DB는 정답 데이터가 아닌, LLM 응답 기반으로 만들어 놓은 테스트 용 DB입니다.**
    -   정답 데이터를 활용해 벡터 DB를 새로 빌드해야 합니다.
    -   `utils/build_store_vector_db.py` 파일의 `JSON_PATH`와 `DB_PATH`를 실제 경로에 맞게 수정한 후, 다음 명령어를 실행하세요.
        ```bash
        python utils/build_store_vector_db.py
        ```
    - 지정한 `DB_PATH`에 맞게 `scoring/nodes/category_nodes.py`의 `similar_partner_search` 도구에서 불러올 벡터 DB의 경로 수정이 필요합니다. 

### 7.3. 실행

-   **API 서버 실행**:
    ```bash
    uvicorn app:app --reload
    ```
    -   서버가 실행되면, 웹 브라우저에서 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 로 접속하여 자동 생성된 API 문서를 확인할 수 있습니다.

-   **일괄 스코어링 실행**:
    ```bash
    python main.py
    ```

-   **성능 평가 실행**:
    ```bash
    python evaluation.py
    ```