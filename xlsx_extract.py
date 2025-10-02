# -*- coding: utf-8 -*-
import os, time, requests
import sqlite3
import pandas as pd
from typing import List, Dict, Optional, Iterable

# ============ 사용자 설정 ============
EXCEL_PATH = "농축수산물 품목 및 등급 코드표.xlsx"
SQLITE_PATH = "kamis_api_list.db"
SQLITE_TABLE = "api_items"

DOWNLOAD_URL = (
    "https://www.kamis.or.kr/customer/board/board_file.do"
    "?brdno=4&brdctsno=424245&brdctsfileno=15636"
)
DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.kamis.or.kr/customer/board/board.do",
}

# 축산물 부류 코드 상수
LIVESTOCK_CATEGORY_CODE = "500"
LIVESTOCK_CATEGORY_NAME = "축산물"

# 시트명
SHEET_BURYU = "부류코드"
SHEET_PUMMOK = "품목코드"
SHEET_PUMJONG = "품종코드"
SHEET_RANK = "등급코드"
SHEET_LIVESTOCK = "축산물 코드"
SHEET_SANMUL = "산물코드"

# ============ 스키마 매핑(최종 DB 스키마) ============
# 최종 DB 컬럼명(=값을 넣을 변수명)
SCHEMA_MAPPING: Dict[str, str] = {
    "부류코드": "category_code",
    "부류명": "category_name",
    "품목코드": "product_code",
    "품목명": "product_name",
    "품종코드": "kind_code",
    "품종명": "kind_name",
    "축산_품종코드": "livestock_kind_code",
    "등급코드(p_productrankcode)": "productrank_code",
    "등급코드(p_graderank)": "graderank_code",
    "등급코드명": "rank_name",
    "등급코드(periodProductList)": "p_periodProductList",
    "등급코드(periodProductName)": "p_periodProductName",
}

# DB 테이블 정의(모두 TEXT) - 축산물 등급명 컬럼 추가
SQLITE_SCHEMA: Dict[str, str] = {
    "category_code": "TEXT NOT NULL",
    "category_name": "TEXT NOT NULL",
    "product_code": "TEXT NOT NULL",
    "product_name": "TEXT NOT NULL",
    "kind_code": "TEXT",
    "kind_name": "TEXT",
    "livestock_kind_code": "TEXT",
    "productrank_code": "TEXT",
    "graderank_code": "TEXT",
    "rank_name": "TEXT",
    "p_periodProductList": "TEXT",
    "p_periodProductName": "TEXT",
}


# ============ 유틸 ============
def _strip_str(x):
    return x.strip() if isinstance(x, str) else x


def _drop_all_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.map(_strip_str).replace("", pd.NA)
    return df.dropna(how="all")


def _merge_duplicate_columns(
    df: pd.DataFrame, base_cols: List[str], suffix: str
) -> pd.DataFrame:
    """중복 컬럼 병합 헬퍼 함수"""
    for col in base_cols:
        dup_col = f"{col}{suffix}"
        if dup_col in df.columns:
            df[col] = df[col].fillna(df[dup_col])
            df = df.drop(dup_col, axis=1)
    return df


def _find_header_row(
    df_raw: pd.DataFrame, required_cols: Iterable[str], max_scan: int = 50
) -> Optional[int]:
    req = set(required_cols)
    for i in range(min(max_scan, len(df_raw))):
        headers = [
            str(v).strip() if pd.notna(v) else "" for v in df_raw.iloc[i].tolist()
        ]
        hdr_set = set(h for h in headers if h)
        if req.issubset(hdr_set):
            return i
    return None


def read_sheet_with_header(
    excel_path: str, sheet_name: str, wanted_cols: List[str], max_scan: int = 50
) -> pd.DataFrame:
    df_raw = pd.read_excel(
        excel_path, sheet_name=sheet_name, header=None, engine="openpyxl"
    )
    header_row = _find_header_row(df_raw, wanted_cols, max_scan=max_scan)
    if header_row is None:
        raise ValueError(
            f"[{sheet_name}] 헤더 행을 찾지 못했습니다. 필요한 컬럼: {wanted_cols}"
        )

    df = pd.read_excel(
        excel_path, sheet_name=sheet_name, header=header_row, engine="openpyxl"
    )
    df.columns = [str(c).strip() for c in df.columns]
    df = df.map(_strip_str)

    for col in wanted_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[wanted_cols]
    return _drop_all_empty_rows(df)


# ============ 시트별 데이터 추출 ============
def extract_buryu(path: str) -> pd.DataFrame:
    return read_sheet_with_header(path, SHEET_BURYU, ["부류코드", "부류명"])


def extract_pummok(path: str) -> pd.DataFrame:
    return read_sheet_with_header(
        path, SHEET_PUMMOK, ["부류코드", "품목코드", "품목명"]
    )


def extract_pumjong(path: str) -> pd.DataFrame:
    wanted = ["품목 코드", "품종코드", "품목명", "품종명"]
    df = read_sheet_with_header(path, SHEET_PUMJONG, wanted)
    # 표준화: "품목 코드" -> "품목코드"
    df = df.rename(columns={"품목 코드": "품목코드"})
    return df[["품목코드", "품목명", "품종코드", "품종명"]]


def extract_rank(path: str) -> pd.DataFrame:
    return read_sheet_with_header(
        path,
        SHEET_RANK,
        ["등급코드(p_productrankcode)", "등급코드(p_graderank)", "등급코드명"],
    )


def extract_livestock(path: str) -> pd.DataFrame:
    wanted = [
        "품목명",
        "품종명",
        "등급명",
        "품목코드(itemcode)",
        "품종코드(kindcode)",
        "등급코드(periodProductList)",
    ]
    df = read_sheet_with_header(path, SHEET_LIVESTOCK, wanted)

    df["부류코드"] = LIVESTOCK_CATEGORY_CODE
    df["부류명"] = LIVESTOCK_CATEGORY_NAME

    rename_map = {
        "품목코드(itemcode)": "품목코드",
        "품종코드(kindcode)": "축산_품종코드",
        "등급코드(periodProductList)": "등급코드(periodProductList)",
        "등급명": "등급코드(periodProductName)",
    }
    df = df.rename(columns=rename_map)

    return df[
        [
            "부류코드",
            "부류명",
            "품목코드",
            "품목명",
            "축산_품종코드",
            "품종명",
            "등급코드(periodProductList)",
            "등급코드(periodProductName)",
        ]
    ]


def extract_sanmul(path: str) -> pd.DataFrame:
    wanted = [
        "품목분류명",
        "품목분류코드",
        "품목명",
        "품목코드",
        "품종명",
        "품종코드",
        "산물등급명",
        "산물등급코드",
    ]
    df = read_sheet_with_header(path, SHEET_SANMUL, wanted)
    # 표준화 rename
    rename_map = {
        "품목분류코드": "부류코드",
        "품목분류명": "부류명",
        "산물등급명": "등급코드명",
        "산물등급코드": "등급코드(p_productrankcode)",
    }
    df = df.rename(columns=rename_map)
    keep = [
        "부류코드",
        "부류명",
        "품목코드",
        "품목명",
        "품종코드",
        "품종명",
        "등급코드(p_productrankcode)",
        "등급코드명",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    return df[keep]


# ============ 데이터 타입 통일 함수 ============
def normalize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """조인 키 컬럼들의 데이터 타입을 문자열로 통일"""
    key_columns = ["부류코드", "품목코드", "품종코드"]

    for col in key_columns:
        if col in df.columns:
            # NaN이 아닌 값들을 문자열로 변환하고 앞뒤 공백 제거
            df[col] = df[col].astype(str).str.strip()
            # 'nan' 문자열을 다시 NaN으로 변환
            df[col] = df[col].replace("nan", pd.NA)

    return df


# ============ 관계형 병합 로직 ============
def merge_hierarchically(path: str) -> pd.DataFrame:
    """계층구조를 따라 관계형 조인으로 데이터 병합"""

    # 1. 각 시트별 데이터 추출
    df_buryu = extract_buryu(path)
    df_pummok = extract_pummok(path)
    df_pumjong = extract_pumjong(path)
    df_rank = extract_rank(path)
    df_livestock = extract_livestock(path)
    df_sanmul = extract_sanmul(path)

    # ★ 데이터 타입 통일 (조인 전 필수!)
    df_buryu = normalize_data_types(df_buryu)
    df_pummok = normalize_data_types(df_pummok)
    df_pumjong = normalize_data_types(df_pumjong)
    df_livestock = normalize_data_types(df_livestock)
    df_sanmul = normalize_data_types(df_sanmul)

    print(f"부류 시트: {len(df_buryu)}행")
    print(f"품목 시트: {len(df_pummok)}행")
    print(f"품종 시트: {len(df_pumjong)}행")
    print(f"등급 시트: {len(df_rank)}행")
    print(
        f"축산물 시트: {len(df_livestock)}행 (부류코드='500', 부류명='축산물' 자동 추가)"
    )
    print(f"산물 시트: {len(df_sanmul)}행")

    # 2. 기본 골격: 부류 → 품목 조인
    result = df_buryu.merge(
        df_pummok, on="부류코드", how="outer", suffixes=("", "_pummok")
    )

    # 부류명 병합 (부류 시트 우선)
    result["부류명"] = result["부류명"].fillna(result.get("부류명_pummok", pd.NA))
    if "부류명_pummok" in result.columns:
        result = result.drop("부류명_pummok", axis=1)

    print(f"부류+품목 조인 후: {len(result)}행")

    # 3. 품종 데이터 조인 (품목코드 기준)
    result = result.merge(
        df_pumjong, on="품목코드", how="outer", suffixes=("", "_pumjong")
    )

    # 품목명 병합 (기존 데이터 우선)
    result = _merge_duplicate_columns(result, ["품목명"], "_pumjong")

    print(f"품종 조인 후: {len(result)}행")

    # 4. 축산물 데이터 조인 (부류코드='500' 기준)
    livestock_merge_cols = ["부류코드", "품목코드"]
    result = result.merge(
        df_livestock, on=livestock_merge_cols, how="outer", suffixes=("", "_livestock")
    )

    # 중복 컬럼 병합 - 축산물 고유 등급 정보는 별도 처리
    result = _merge_duplicate_columns(
        result, ["부류명", "품목명", "품종명"], "_livestock"
    )

    print(f"축산물 조인 후: {len(result)}행")

    # 5. 산물 데이터 조인
    sanmul_merge_cols = ["부류코드", "품목코드", "품종코드"]
    result = result.merge(
        df_sanmul, on=sanmul_merge_cols, how="outer", suffixes=("", "_sanmul")
    )

    # 중복 컬럼 병합
    result = _merge_duplicate_columns(
        result, ["부류명", "품목명", "품종명", "등급코드명"], "_sanmul"
    )

    print(f"산물 조인 후: {len(result)}행")

    # 6. 등급 데이터 조인
    if not df_rank.empty:
        non_livestock_mask = result["부류코드"] != LIVESTOCK_CATEGORY_CODE
        livestock_data = result[~non_livestock_mask].copy()
        non_livestock_data = result[non_livestock_mask].copy()

        if not non_livestock_data.empty:
            # cross join 전에 충돌할 수 있는 컬럼들을 미리 삭제합니다.
            # 이렇게 하면 merge 후에 _x, _y 같은 접미사가 붙는 것을 방지할 수 있습니다.
            cols_to_drop = [
                col for col in df_rank.columns if col in non_livestock_data.columns
            ]
            non_livestock_data_cleaned = non_livestock_data.drop(columns=cols_to_drop)

            # 정리된 데이터프레임으로 cross join을 수행합니다.
            non_livestock_with_rank = non_livestock_data_cleaned.merge(
                df_rank, how="cross"
            )
        else:
            non_livestock_with_rank = pd.DataFrame()  # 비어있는 경우를 위한 처리

        # 축산물 데이터와 비축산물+등급 데이터 합치기
        # 축산물 데이터에는 일반 등급 컬럼을 빈 값으로 추가
        if not livestock_data.empty:
            for col in df_rank.columns:
                if col not in livestock_data.columns:
                    livestock_data[col] = pd.NA

        # 최종 결합
        if not livestock_data.empty and not non_livestock_with_rank.empty:
            result = pd.concat(
                [livestock_data, non_livestock_with_rank], ignore_index=True
            )
        elif not livestock_data.empty:
            result = livestock_data
        elif not non_livestock_with_rank.empty:
            result = non_livestock_with_rank
        else:
            result = result  # 원본 유지

        print(f"등급 조인 후: {len(result)}행 (축산물은 고유 등급 체계 사용)")

    # 7. 데이터 정제
    result = result.map(_strip_str).replace("", pd.NA)
    result = result.drop_duplicates().reset_index(drop=True)

    print(f"중복 제거 후 최종: {len(result)}행")

    return result


# ============ 최종 스키마 매핑 ============
def map_to_final_schema(df: pd.DataFrame) -> pd.DataFrame:
    """최종 DB 스키마로 매핑"""
    out = pd.DataFrame()

    # SCHEMA_MAPPING에 따라 컬럼 매핑
    for src_col, dst_col in SCHEMA_MAPPING.items():
        out[dst_col] = (
            df[src_col].astype("string")
            if src_col in df.columns
            else pd.Series(dtype="string")
        )

    # 전체 string 통일 + 중복 제거
    for c in out.columns:
        out[c] = out[c].astype("string")

    return out.drop_duplicates().reset_index(drop=True)


def process_kamis_data(
    excel_path: str = EXCEL_PATH,
    sqlite_path: str = SQLITE_PATH,
    table_name: str = SQLITE_TABLE,
    auto_download: bool = True,
    return_dataframe: bool = False,
) -> Optional[pd.DataFrame]:
    """
    KAMIS 데이터 처리 메인 함수

    Args:
        excel_path: 엑셀 파일 경로 (기본값: EXCEL_PATH)
        sqlite_path: SQLite DB 경로 (기본값: SQLITE_PATH)
        table_name: 테이블명 (기본값: SQLITE_TABLE)
        auto_download: 자동 다운로드 여부 (기본값: True)
        return_dataframe: DataFrame 반환 여부 (기본값: False)

    Returns:
        return_dataframe=True인 경우 처리된 DataFrame 반환, 아니면 None

    Examples:
        # 기본 사용 (DB만 생성)
        >>> process_kamis_data()

        # DataFrame 받아서 추가 처리
        >>> df = process_kamis_data(return_dataframe=True)
        >>> filtered = df[df['category_name'] == '과일류']

        # 커스텀 경로
        >>> process_kamis_data(
        ...     excel_path="custom.xlsx",
        ...     sqlite_path="custom.db",
        ...     auto_download=False
        ... )
    """
    print("=== 관계형 병합 방식으로 데이터 처리 시작 ===")

    if auto_download:
        download_if_needed(excel_path, DOWNLOAD_URL, DOWNLOAD_HEADERS)

    # 1. 관계형 조인으로 데이터 병합
    merged_data = merge_hierarchically(excel_path)

    # 2. 최종 스키마로 매핑
    final_data = map_to_final_schema(merged_data)

    # 3. SQLite에 저장
    load_to_sqlite(final_data, sqlite_path, table_name)

    print(f"\n[완료] SQLite 적재 완료!")
    print(f"- 파일: {sqlite_path}")
    print(f"- 테이블: {table_name}")
    print(f"- 최종 행수: {len(final_data)}")

    if return_dataframe:
        return final_data
    return None


# ============ SQLite 관련 ============
def create_sqlite_table(conn: sqlite3.Connection, table: str, schema: Dict[str, str]):
    cols = ", ".join([f'"{k}" {v}' for k, v in schema.items()])
    conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({cols});')


def create_indexes(conn: sqlite3.Connection, table: str):
    idx_cols = [
        ("idx_category_code", "category_code"),
        ("idx_product_code", "product_code"),
        ("idx_kind_code", "kind_code"),
        ("idx_livestock_kind_code", "livestock_kind_code"),
        ("idx_productrank_code", "productrank_code"),
        ("idx_graderank_code", "graderank_code"),
    ]
    for name, col in idx_cols:
        conn.execute(f'CREATE INDEX IF NOT EXISTS {name} ON "{table}"("{col}");')


def load_to_sqlite(df: pd.DataFrame, sqlite_path: str, table: str):
    # ✅ 필수 컬럼 보정
    required = ["category_code", "category_name", "product_code", "product_name"]
    df = df.copy()
    for c in required:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    # 빈 문자열을 NA로 치환 후 드롭
    df[required] = df[required].replace("", pd.NA)
    before = len(df)
    df = df.dropna(subset=required)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[warn] NOT NULL 위반 방지를 위해 {dropped}행 제거")

    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(f'DROP TABLE IF EXISTS "{table}";')
        create_sqlite_table(conn, table, SQLITE_SCHEMA)
        df.to_sql(table, conn, if_exists="append", index=False)
        create_indexes(conn, table)


def download_if_needed(
    path: str, url: str, headers: Dict[str, str] | None = None, max_age_hours: int = 24
):
    """
    path가 없거나, 마지막 수정 시간이 max_age_hours 이상 지났으면 url에서 받아 저장한다.
    """
    # 파일이 이미 존재하면 수정된 시간 확인
    if os.path.exists(path):
        file_age_hours = (time.time() - os.path.getmtime(path)) / 3600
        if file_age_hours < max_age_hours:
            # 24시간 이내라면 재다운로드 안 함
            print(
                f"[download] 이미 최신 파일이 존재합니다. ({file_age_hours:.1f}시간 경과)"
            )
            return
        else:
            print(
                f"[download] 파일이 {file_age_hours:.1f}시간 경과. 새로 다운로드합니다."
            )

    # 다운로드 실행
    h = headers or {}
    r = requests.get(url, headers=h, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    print(f"[download] Saved: {path}")


def main():
    """스크립트 직접 실행 시 호출되는 함수"""
    process_kamis_data()


if __name__ == "__main__":
    main()
