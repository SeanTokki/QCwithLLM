import json, os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import random

# ---------- 설정 ----------
JSON_PATH   = "./data/final/scored_inner_store_data.json"
DB_DIR      = "chroma_store"
COLLECTION  = "stores"
MODEL_NAME  = "dragonkue/multilingual-e5-small-ko"   # 384-dim
BATCH       = 64
TOP_K       = 5
# --------------------------

def build_plain_text(rec: dict) -> str:
    def take_menu(items, n=10):
        if items == None:
            return ""
        names = []
        for item in items:
            if item["is_representative"] == True:
                names.append(item["name"])
        
        if len(names) >= n:
            return ", ".join(names[:n])
        else:
            require = n - len(names)
            for item in items:
                if item["is_representative"] == False:
                    names.append(item["name"])
                    require -= 1
                if require == 0:
                    break
            return ", ".join(names)

    menu_names = take_menu(rec.get("menu_list", []))
    review_kw = ", ".join(list(rec.get("review_category", {}).keys()))
    reviews = " ".join(c["comment"] for c in rec.get("review_list", [])[:5])

    tpl = (
        f"passage: 카테고리: {rec.get('category','')}\n"
        f"대표메뉴: {menu_names}\n" # 대표메뉴 + 일반메뉴 최대 10개
        f"리뷰키워드: {review_kw}\n"
        f"최근리뷰: {reviews}" # 최근 리뷰 최대 5개
    )
    return tpl.strip()

def build():
    # 데이터 로드
    records = json.load(open(JSON_PATH, encoding="utf-8"))

    # 임베딩 함수 준비
    model      = SentenceTransformer(MODEL_NAME)
    embed_fn   = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME,  # Chroma wrapper용
        device="cuda" if model.device.type == "cuda" else "cpu",
    )

    # Chroma 설정
    client     = chromadb.PersistentClient(path=DB_DIR)
    coll       = client.get_or_create_collection(COLLECTION, embedding_function=embed_fn)

    # 벡터 추가 (id는 문자열이어야 함)
    texts, meta, ids = [], [], []
    for rec in records:
        texts.append(build_plain_text(rec))
        meta.append(
            {
                "naver_id": rec["naver_id"], 
                "name": rec["name"], 
                "top_category": rec["top_category"],
                "sub_category": rec["sub_category"],
                "score": rec["score"],
                "reasoning": rec["reasoning"]
            }
        )
        ids.append(str(rec["naver_id"]))

    # 대량 입력은 upsert 메서드
    for i in range(0, len(texts), BATCH):
        coll.upsert(
            ids   = ids[i:i+BATCH],
            documents = texts[i:i+BATCH],
            metadatas = meta[i:i+BATCH],
        )

    print(f"✅ {len(texts):,}개 레코드를 Chroma(DB='{DB_DIR}', collection='{COLLECTION}')에 저장 완료")
    
    return client

def test():
    client     = chromadb.PersistentClient(path=DB_DIR)
    coll       = client.get_or_create_collection(COLLECTION)
    
    with open("./data/final/crawled_new_store_data.json", "r", encoding="utf-8") as f:
        one_store_data = random.choice(json.load(f))
    
    test_query = build_plain_text(one_store_data)
    print("[Test Query]")
    print(test_query)
    print("============================================================")

    res = coll.query(
        query_texts = [f"query: {test_query}"],
        n_results   = TOP_K,
        include     = ["distances", "metadatas"],
    )

    print("[Search Results]")
    for rank, (m, d) in enumerate(zip(res["metadatas"][0], res["distances"][0]), 1):
        print(f"{rank}. {m['name']} (naver_id={m['naver_id']})  score={1-d:.3f}")

if __name__ == "__main__":
    build()
    # test()