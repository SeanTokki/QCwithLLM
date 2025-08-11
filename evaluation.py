import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import cohen_kappa_score
import json
import matplotlib.pyplot as plt


def compute_metrics(df: pd.DataFrame, tau: float):
    gt   = df["ground_truth"].to_numpy(float)
    pred = df["prediction"].to_numpy(float)
    
    # 0.1점 단위로 묶기
    gt_bin   = np.round(df["ground_truth"] * 10).astype(int)
    pred_bin = np.round(df["prediction"] * 10).astype(int)

    # 1) kappa score
    kappa = cohen_kappa_score(gt_bin, pred_bin, weights="quadratic")

    # 2) MAE
    mae = mean_absolute_error(gt, pred)

    # 3) τ-Accuracy
    acc_tau = (np.abs(gt - pred) <= tau).mean()
    

    return kappa, mae, acc_tau

def load_json_as_df(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        return pd.json_normalize(json.load(f))

def make_comparison_table(gt_df: pd.DataFrame,
                          pred_df: pd.DataFrame,
                          score_cols=None,
                          out_path: str | None = None) -> pd.DataFrame:
    if score_cols is None:
        score_cols = sorted(
            c for c in gt_df.columns if c.endswith("_score") and c in pred_df.columns
        )

    # GT / Pred 각각 접미사 붙여서 조인
    gt_part   = gt_df[["naver_id", *score_cols]].set_index("naver_id").add_suffix("_gt")
    pred_part = pred_df[["naver_id", *score_cols]].set_index("naver_id").add_suffix("_pred")

    tbl = gt_part.join(pred_part, how="inner")

    # Error 컬럼 추가
    for col in score_cols:
        tbl[f"{col}_err"] = tbl[f"{col}_pred"] - tbl[f"{col}_gt"]

    # ===== 선택적으로 파일로 저장 =====
    if out_path:
        ext = out_path.rsplit(".", 1)[-1].lower()
        if ext == "csv":
            tbl.to_csv(out_path, encoding="utf-8-sig")
        elif ext in {"xlsx", "xls"}:
            tbl.to_excel(out_path, index=True)
        elif ext in {"html", "htm"}:
            tbl.to_html(out_path, index=True)
        else:
            raise ValueError("지원하지 않는 확장자입니다: " + ext)

    return tbl

def dict_to_list(ipath: str, opath: str):
    with open(ipath, "r", encoding="utf-8") as f:
        data: dict = json.load(f)
    
    new_data = []
    for item in data.values():
        new_data.append(item)
    
    with open(opath, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 1) 데이터 로드 (list of dict 형태여야 함)
    dict_to_list("./data/test/scores.json", "./data/test/scores_list.json")
    gt_df   = load_json_as_df("./data/examples/example_scores.json")
    pred_df = load_json_as_df("./data/test/scores_list.json")

    # 2) ground-truth에 있는 매장만 필터링하여 병합
    df = (gt_df
          .merge(pred_df, on="naver_id", suffixes=("_gt", "_pred"))
          .set_index("naver_id"))

    # 3) 평가가 필요한 컬럼 목록
    target_cols = ["pos_score", "cat_score", "inn_score", "seat_score", "add_score", "tot_score"]

    print("====== Evaluation Result (Total Score τ = 0.5) ======")
    for col in target_cols:
        sub = df[[f"{col}_gt", f"{col}_pred"]].rename(
            columns={f"{col}_gt": "ground_truth", f"{col}_pred": "prediction"}
        )
        if col == "tot_score":
            kappa, mae, acc = compute_metrics(sub, tau=0.5)
        else:
            kappa, mae, acc = compute_metrics(sub, tau=0.1)
        
        print(f"{col:>10} | κ={kappa:5.3f}  MAE={mae:5.3f}  Acc={acc*100:5.1f}%")
    
    #4) 표로 저장
    comp_tbl = make_comparison_table(gt_df, pred_df,
                                     score_cols=["pos_score", "cat_score", "inn_score",
                                                 "seat_score", "add_score", "tot_score"],
                                     out_path="comparison_table.csv")