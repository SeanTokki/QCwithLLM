from shapely.geometry import Point, Polygon
from shapely import wkt
import pandas as pd

from scoring.schema import *

async def load_polygons(file_path: str, name_col: str, polygon_col: str) -> Dict[str, Polygon]:
    polygons = {}
    try:
        df = pd.read_csv(file_path)
        if name_col not in df.columns or polygon_col not in df.columns:
            print(f"'{file_path}'에 '{name_col}' 또는 '{polygon_col}' 컬럼이 없습니다.")
            return {}

        for _, row in df.iterrows():
            try:
                name = row[name_col]
                poly = wkt.loads(row[polygon_col])
                polygons[name] = poly
            except Exception as e:
                continue # 실패한 폴리곤은 건너뛰기
    except FileNotFoundError:
        print(f"오류: Polygon 파일 '{file_path}'를 찾을 수 없습니다. 경로를 확인해주세요.")
    except Exception as e:
        print(f"오류: Polygon 파일 '{file_path}' 로드 중 예상치 못한 오류 발생: {e}")
    return polygons

async def get_position_result(store_data: Dict[str, Any])-> Optional[PositionResult]:
    point = Point(store_data["gps_longitude"], store_data["gps_latitude"])
    new_hot_keywords = ["삼성역", "코엑스", "익선동", "샤로수길", "송리단길", "해방촌", "후암동", "서촌"]
    hotplace_polys = await load_polygons("./data/polygons/seoul_hotspots_polygons.csv", "location", "polygon_str")
    campus_polys = await load_polygons("./data/polygons/campus_polygons.csv", "campus_name", "polygon_str")
    store_name = store_data["name"]
    
    # 평가에 필요한 폴리곤을 불러오지 못했으므로 None 반환
    if not hotplace_polys or not campus_polys:
        return None
    
    # 핫플레이스 안에 위치하면 5점
    for name, poly in hotplace_polys.items():
        if point.within(poly):
            return PositionResult(
                name=store_name,
                score=5.0,
                reason="핫플레이스 안에 위치"
            )
    
    # 신규 핫플레이스 안에 위치하면 4점
    # 현재는 주소 안에 키워드로 판별
    for keyword in new_hot_keywords:
        if keyword in store_data["address"]:
            return PositionResult(
                name=store_name,
                score=4.0,
                reason="신규 핫플레이스 안에 위치"
            )
    
    # 대학가 안에 위치하면 4점
    for name, poly in campus_polys.items():
        if point.within(poly):
            return PositionResult(
                name=store_name,
                score=4.0,
                reason="대학가에 위치"
            )
    
    if store_data["distance_from_subway"]:
        # 지하철 역과의 거리가 도보 15분 이내(750m)면 3점
        if store_data["distance_from_subway"] <= 750:
            return PositionResult(
                name=store_name,
                score=3.0,
                reason="역에서 도보 15분 이내"
            )
        # 지하철 역과의 거리가 도보 25분 이내(1250m)면 2점
        elif store_data["distance_from_subway"] <= 1250:
            return PositionResult(
                name=store_name,
                score=2.0,
                reason="역에서 도보 25분 이내"
            )
        # 지하철 역과의 거리가 도보 25분 초과(버스 환승 필요)면 1점
        else:
            return PositionResult(
                name=store_name,
                score=1.0,
                reason="역에서 버스 환승 필요"
            )
    
    else:
        # 지하철 역과 멀기 때문에 지하철 역과의 거리 정보가 없다고 생각하고 1점
        return PositionResult(
                name=store_name,
                score=1.0,
                reason="역에서 버스 환승 필요"
        )