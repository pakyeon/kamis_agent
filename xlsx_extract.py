# -*- coding: utf-8 -*-
import os, time, requests
import tempfile
import shutil
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

DB_MAX_AGE_HOURS = 24  # 로컬 SQLite DB가 이 값(시간)보다 오래되면 갱신

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


def download_to_path(url: str, dest_path: str, headers: Dict[str, str] | None = None):
    h = headers or {}
    r = requests.get(url, headers=h, timeout=60)
    r.raise_for_status()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(r.content)
    print(f"[download] Saved (temp): {dest_path}")


def should_update_db(sqlite_path: str, max_age_hours: int) -> bool:
    """
    로컬 SQLite DB 최신성(TTL) 기준으로 갱신 필요 여부 판단.
    - DB가 없으면 True
    - DB의 mtime이 max_age_hours를 넘으면 True
    - 그 외 False
    """
    if not os.path.exists(sqlite_path):
        print("[update] DB가 존재하지 않습니다 → 갱신합니다.")
        return True

    db_mtime = os.path.getmtime(sqlite_path)
    age_hours = (time.time() - db_mtime) / 3600.0

    if age_hours > max_age_hours:
        print(f"[update] DB가 {age_hours:.1f}h 경과(임계 {max_age_hours}h) → 갱신.")
        return True

    print(
        f"[update] DB가 최신입니다({age_hours:.1f}h 경과 ≤ {max_age_hours}h). 재생성 스킵."
    )
    return False


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


# ============ Lookup 기반 병합 헬퍼 함수 ============
def merge_with_lookup(
    data_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    merge_keys: list,
    lookup_cols: dict,
) -> pd.DataFrame:
    """
    데이터 손실 없이 lookup 정보를 병합하는 함수

    Args:
        data_df: 메인 데이터 프레임
        lookup_df: 조회용 데이터 프레임
        merge_keys: 조인 키 리스트
        lookup_cols: {원본컬럼명: 대상컬럼명} 매핑
    """
    result = data_df.copy()

    # lookup 데이터와 left join
    merged = result.merge(
        lookup_df[merge_keys + list(lookup_cols.keys())],
        on=merge_keys,
        how="left",
        suffixes=("", "_lookup"),
    )

    # lookup 값으로 빈 값 채우기
    for src_col, dst_col in lookup_cols.items():
        lookup_col = (
            f"{src_col}_lookup" if f"{src_col}_lookup" in merged.columns else src_col
        )
        if dst_col not in merged.columns:
            merged[dst_col] = merged[lookup_col]
        else:
            merged[dst_col] = merged[dst_col].fillna(merged[lookup_col])

    # lookup 접미사 컬럼 제거
    cols_to_drop = [c for c in merged.columns if c.endswith("_lookup")]
    merged = merged.drop(columns=cols_to_drop)

    return merged


# ============ 관계형 병합 로직 (개선 버전) ============
def merge_hierarchically(path: str) -> pd.DataFrame:
    """
    개선된 병합 로직: 데이터 손실 최소화
    각 시트를 독립적으로 처리한 후 통합
    """

    # 1. 각 시트별 데이터 추출
    df_buryu = extract_buryu(path)
    df_pummok = extract_pummok(path)
    df_pumjong = extract_pumjong(path)
    df_rank = extract_rank(path)
    df_livestock = extract_livestock(path)
    df_sanmul = extract_sanmul(path)

    # 데이터 타입 통일
    df_buryu = normalize_data_types(df_buryu)
    df_pummok = normalize_data_types(df_pummok)
    df_pumjong = normalize_data_types(df_pumjong)
    df_livestock = normalize_data_types(df_livestock)
    df_sanmul = normalize_data_types(df_sanmul)

    print(f"부류 시트: {len(df_buryu)}행")
    print(f"품목 시트: {len(df_pummok)}행")
    print(f"품종 시트: {len(df_pumjong)}행")
    print(f"등급 시트: {len(df_rank)}행")
    print(f"축산물 시트: {len(df_livestock)}행")
    print(f"산물 시트: {len(df_sanmul)}행")

    # 2. 일반 품목 데이터 처리 (산물 시트 기반)
    print("\n[일반 품목 처리]")
    general_items = df_sanmul.copy()
    print(f"산물 시트 원본: {len(general_items)}행")

    # 부류명 lookup
    general_items = merge_with_lookup(
        general_items, df_buryu, ["부류코드"], {"부류명": "부류명"}
    )

    # 품목명 lookup
    general_items = merge_with_lookup(
        general_items,
        df_pummok[["품목코드", "품목명"]].drop_duplicates(),
        ["품목코드"],
        {"품목명": "품목명"},
    )

    # 품종명 lookup
    general_items = merge_with_lookup(
        general_items,
        df_pumjong[["품목코드", "품종코드", "품종명"]].drop_duplicates(),
        ["품목코드", "품종코드"],
        {"품종명": "품종명"},
    )

    # 등급 정보 lookup
    if not df_rank.empty:
        general_items = merge_with_lookup(
            general_items,
            df_rank,
            ["등급코드(p_productrankcode)"],
            {
                "등급코드(p_graderank)": "등급코드(p_graderank)",
                "등급코드명": "등급코드명",
            },
        )

    # 축산물 컬럼 추가 (빈 값)
    general_items["축산_품종코드"] = pd.NA
    general_items["등급코드(periodProductList)"] = pd.NA
    general_items["등급코드(periodProductName)"] = pd.NA

    print(f"일반 품목 처리 후: {len(general_items)}행")

    # 3. 축산물 데이터 처리
    print("\n[축산물 처리]")
    livestock_items = df_livestock.copy()
    print(f"축산물 시트 원본: {len(livestock_items)}행")

    # 품목명 lookup
    livestock_items = merge_with_lookup(
        livestock_items,
        df_pummok[["품목코드", "품목명"]].drop_duplicates(),
        ["품목코드"],
        {"품목명": "품목명"},
    )

    # 일반 등급 컬럼 추가 (빈 값)
    livestock_items["품종코드"] = pd.NA
    livestock_items["등급코드(p_productrankcode)"] = pd.NA
    livestock_items["등급코드(p_graderank)"] = pd.NA
    livestock_items["등급코드명"] = pd.NA

    print(f"축산물 처리 후: {len(livestock_items)}행")

    # 4. 산물과 축산물 외의 기본 품목/품종 데이터 처리
    print("\n[기본 품목/품종 처리]")

    # 품목 기반 데이터 생성
    base_items = df_pummok.copy()

    # 부류명 lookup
    base_items = merge_with_lookup(
        base_items, df_buryu, ["부류코드"], {"부류명": "부류명"}
    )

    # 품종이 있는 품목의 경우 품종별로 확장
    items_with_kind = base_items.merge(
        df_pumjong[["품목코드", "품종코드", "품종명"]], on="품목코드", how="inner"
    )

    # 품종이 없는 품목
    items_without_kind = base_items[
        ~base_items["품목코드"].isin(items_with_kind["품목코드"])
    ].copy()
    items_without_kind["품종코드"] = pd.NA
    items_without_kind["품종명"] = pd.NA

    # 합치기
    base_items = pd.concat([items_with_kind, items_without_kind], ignore_index=True)

    # 축산물과 산물에 이미 포함된 조합 제거
    sanmul_combos = set(
        df_sanmul[["부류코드", "품목코드", "품종코드"]]
        .dropna(subset=["품목코드"])
        .apply(
            lambda x: f"{x['부류코드']}|{x['품목코드']}|{str(x['품종코드'])}", axis=1
        )
    )
    livestock_combos = set(
        df_livestock[["부류코드", "품목코드"]].apply(
            lambda x: f"{x['부류코드']}|{x['품목코드']}|", axis=1
        )
    )

    base_items["_combo"] = base_items.apply(
        lambda x: f"{x['부류코드']}|{x['품목코드']}|{str(x['품종코드'])}", axis=1
    )
    base_items = base_items[
        ~base_items["_combo"].isin(sanmul_combos)
        & ~base_items["_combo"].isin(livestock_combos)
    ].drop("_combo", axis=1)

    # 빈 컬럼 추가
    base_items["축산_품종코드"] = pd.NA
    base_items["등급코드(p_productrankcode)"] = pd.NA
    base_items["등급코드(p_graderank)"] = pd.NA
    base_items["등급코드명"] = pd.NA
    base_items["등급코드(periodProductList)"] = pd.NA
    base_items["등급코드(periodProductName)"] = pd.NA

    print(f"기본 품목/품종 데이터: {len(base_items)}행")

    # 5. 모든 데이터 통합
    print("\n[최종 통합]")

    # 컬럼 순서 통일
    final_columns = [
        "부류코드",
        "부류명",
        "품목코드",
        "품목명",
        "품종코드",
        "품종명",
        "축산_품종코드",
        "등급코드(p_productrankcode)",
        "등급코드(p_graderank)",
        "등급코드명",
        "등급코드(periodProductList)",
        "등급코드(periodProductName)",
    ]

    # 각 데이터프레임의 컬럼 통일
    for df in [general_items, livestock_items, base_items]:
        for col in final_columns:
            if col not in df.columns:
                df[col] = pd.NA

    # 순서대로 선택
    general_items = general_items[final_columns]
    livestock_items = livestock_items[final_columns]
    base_items = base_items[final_columns]

    # 통합
    result = pd.concat([general_items, livestock_items, base_items], ignore_index=True)

    print(f"통합 후: {len(result)}행")

    # 6. 데이터 정제
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
    db_max_age_hours: int | None = None,
    download_url: str | None = None,
    download_headers: dict | None = None,
):
    """
    KAMIS 데이터 처리 메인 함수 (로컬 SQLite DB 최신성 기반 갱신)
    """
    import os
    import time
    import tempfile

    # --- 내부 헬퍼: 로컬 DB TTL 기반 업데이트 필요 여부 판단 ---
    def _should_update_db(_sqlite_path: str, _max_age_hours: int) -> bool:
        if not os.path.exists(_sqlite_path):
            print("[update] DB가 존재하지 않습니다 → 갱신합니다.")
            return True
        db_mtime = os.path.getmtime(_sqlite_path)
        age_hours = (time.time() - db_mtime) / 3600.0
        if age_hours > _max_age_hours:
            print(
                f"[update] DB가 {age_hours:.1f}h 경과(임계 {_max_age_hours}h) → 갱신."
            )
            return True
        print(
            f"[update] DB가 최신입니다({age_hours:.1f}h 경과 ≤ {_max_age_hours}h). 재생성 스킵."
        )
        return False

    # --- 파라미터/전역 기본값 해석 ---
    # TTL
    if db_max_age_hours is None:
        try:
            # 전역 상수가 있으면 사용
            _db_ttl = DB_MAX_AGE_HOURS
        except NameError:
            _db_ttl = 24  # 기본 24시간
    else:
        _db_ttl = int(db_max_age_hours)

    # 다운로드 URL/헤더
    if download_url is None:
        download_url = DOWNLOAD_URL
    if download_headers is None:
        download_headers = DOWNLOAD_HEADERS

    print("=== 개선된 병합 방식으로 데이터 처리 시작 ===")

    temp_dir = None
    excel_path_in_use = excel_path

    try:
        if auto_download:
            needs_update = _should_update_db(sqlite_path, _db_ttl)
            if not needs_update:
                # 최신이면 아무 것도 하지 않음
                return None

            print("[update] 로컬 DB TTL 초과 → 갱신합니다.")
            # 임시 디렉터리 생성 및 다운로드
            temp_dir = tempfile.TemporaryDirectory(prefix="kamis_")
            excel_path_in_use = os.path.join(temp_dir.name, "kamis_codes.xlsx")

            # 심플 다운로드 (전역/외부의 download_to_path 함수를 사용)
            download_to_path(download_url, excel_path_in_use, download_headers)

        else:
            # 수동 경로 모드: 경로 검증
            if not os.path.exists(excel_path_in_use):
                raise FileNotFoundError(f"엑셀 파일이 없습니다: {excel_path_in_use}")

        # 1) 관계형 병합
        merged_data = merge_hierarchically(excel_path_in_use)

        # 2) 최종 스키마 매핑
        final_data = map_to_final_schema(merged_data)

        # 3) SQLite 적재
        load_to_sqlite(final_data, sqlite_path, table_name)

        print(f"\n[완료] SQLite 적재 완료!")
        print(f"- 파일: {sqlite_path}")
        print(f"- 테이블: {table_name}")
        print(f"- 최종 행수: {len(final_data)}")

        if return_dataframe:
            return final_data
        return None

    finally:
        # auto_download로 받은 임시 파일/디렉터리 정리
        if temp_dir is not None:
            try:
                temp_dir.cleanup()
                print("[cleanup] 임시 파일/디렉터리 삭제 완료.")
            except Exception as e:
                print(f"[cleanup][warn] 임시 파일 삭제 중 경고: {e}")


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
