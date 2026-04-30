"""Korean UI copy constants used by v3 views and components."""

APP_NAME = "Process Factor Finder v3"
APP_SUBTITLE = "공정 X-Y 유의 후보 탐색 워크벤치"

PAGE_LABELS = {
    "Home": "홈",
    "Data": "데이터",
    "Merge": "병합",
    "Analysis Plan": "분석 계획",
    "Run": "분석 실행",
    "Results": "결과",
    "Settings": "설정",
}

PAGE_SUBTITLES = {
    "Home": "공정 X와 품질 Y를 병합하고, 우선 확인할 유의 후보를 빠르게 좁혀봅니다.",
    "Data": "X/Y 파일을 업로드합니다. 미리보기는 head(100), 최대 80컬럼으로 제한합니다.",
    "Merge": "X 내부 병합, Y 내부 병합, 최종 X/Y 병합을 진행합니다.",
    "Analysis Plan": "분석 모드와 후보 범위를 정합니다. 무거운 설정은 접어 두어 화면을 가볍게 유지합니다.",
    "Run": "분석 중에는 표와 그래프를 다시 그리지 않고 진행 상태만 표시합니다.",
    "Results": "상위 후보를 확인하고, 필요한 그래프와 분석 페이지 이미지만 저장합니다.",
    "Settings": "현재 설정값을 확인합니다. 상세 변경은 config.yaml 기준으로 관리합니다.",
}

BUTTONS = {
    "upload_save": "업로드 저장",
    "merge_run": "병합 실행",
    "plan_save": "분석 계획 저장",
    "analysis_run": "분석 실행",
    "filter_apply": "필터 적용",
    "save_graphs": "Top 유의인자 그래프 저장",
    "save_capture": "유의인자 분석 페이지 캡쳐 저장",
    "reset_session": "세션 초기화",
    "go_data": "데이터 업로드로 이동",
    "go_merge": "병합으로 이동",
    "go_plan": "분석 계획으로 이동",
    "go_run": "분석 실행으로 이동",
    "go_results": "결과 보기",
}

LABELS = {
    "navigation": "메뉴",
    "status": "상태",
    "x_files": "X 파일",
    "y_files": "Y 파일",
    "merge": "병합",
    "result": "결과",
    "recent_run": "최근 실행",
    "rows": "행 수",
    "columns": "컬럼 수",
    "file": "파일",
    "analysis_mode": "분석 모드",
    "preset": "프리셋",
    "include_categorical": "범주형 포함",
    "top_n_per_y": "Y별 후보 수",
    "detailed_top_n": "정밀분석 후보 수",
    "final_top_n": "최종 그래프 저장 개수",
    "pair_type": "조합 유형",
    "score_threshold": "점수 기준",
    "top_n_display": "표시할 후보 수",
    "x_search": "X 검색",
    "y_search": "Y 검색",
    "candidate_select": "상세 후보 선택",
    "show_graph": "그래프 보기",
    "save_folder": "저장 폴더",
}

PAIR_TYPE_LABELS = {
    "numeric_numeric": "숫자-숫자",
    "categorical_numeric": "범주-숫자",
    "numeric_categorical": "숫자-범주",
    "categorical_categorical": "범주-범주",
}

PAIR_TYPE_HELP = {
    "numeric_numeric": "두 숫자형 값이 함께 증가하거나 감소하는 패턴을 확인합니다.",
    "categorical_numeric": "X의 값 또는 조건에 따라 Y의 평균/분포 차이가 있는지 확인합니다.",
    "numeric_categorical": "Y의 판정/범주에 따라 X 값의 분포가 달라지는지 확인합니다.",
    "categorical_categorical": "두 범주형 변수의 조합이 독립적이지 않은지 확인합니다.",
}

EMPTY_STATES = {
    "no_upload": "X/Y 파일을 업로드하면 다음 단계에서 Key 병합을 진행할 수 있습니다.",
    "merge_need_upload": "먼저 데이터 화면에서 X/Y 파일을 업로드하세요.",
    "plan_need_merge": "먼저 병합 화면에서 X/Y 최종 병합을 완료하세요.",
    "plan_no_y": "Y 후보 컬럼이 없습니다. 병합 Key 또는 Y 파일을 확인하세요.",
    "run_need_plan": "먼저 분석 계획을 저장하세요.",
    "result_empty": "아직 분석 결과가 없습니다. 분석 실행 화면에서 분석을 실행하세요.",
    "no_display_data": "표시할 데이터가 없습니다.",
    "no_result": "아직 결과가 없습니다.",
}

MODE_LABELS = {
    "full_scan": "전체 후보 빠른 탐색",
    "single_y": "단일 Y 정밀 분석",
}

RISK_LABELS = {
    "low": "낮음",
    "medium": "보통",
    "high": "높음",
    "very high": "매우 높음",
}

NOT_CAUSAL = "본 결과는 우선 확인 후보를 제시하는 것이며, 공정 원인 판단은 별도 검증이 필요합니다."
GRAPH_CAPTION = "우선 확인 후보이며 인과관계 확정은 아닙니다."
LARGE_DATA_NOTICE = "대용량 데이터에서는 화면 표시를 제한하고, 필요한 순간에만 데이터를 읽어 UI 버벅임을 줄입니다."
