# Process Factor Finder v3

Streamlit 기반 공정 X-Y 유의 후보 탐색 워크벤치입니다.

v3는 v2를 계속 패치하지 않고 새 구조로 다시 만든 프로젝트입니다. `app.py`는 라우터만 담당하고, 분석/저장/그래프 로직은 `core/`, `services/`, `views/`로 분리했습니다.

## 실행

```powershell
cd C:\Users\admin\Desktop\python\process_factor_finder_v3
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

목표 Python 버전은 3.12입니다.

## 화면 흐름

1. `홈`: 앱 목적과 현재 상태 확인
2. `데이터`: X/Y 파일 업로드
3. `병합`: X 내부 병합, Y 내부 병합, 최종 X/Y 병합
4. `분석 계획`: 분석 모드, 프리셋, 후보 범위 설정
5. `분석 실행`: 진행률만 표시하며 분석 실행
6. `결과`: Top 후보, 상세 설명, 필요한 그래프만 on-demand 표시
7. `설정`: 현재 config 확인

## 분석 방식

- 전체 pair를 모두 정밀분석하지 않습니다.
- 먼저 matrix screening으로 후보를 빠르게 줄입니다.
- Top 후보만 원본 데이터 기준 detailed analysis를 수행합니다.
- `screening_score`는 후보 선별용입니다.
- `final_score`는 detailed analysis 결과 기반으로만 계산합니다.
- 점수 계산은 `core/scoring.py`에서만 관리합니다.

## 지원 pair type

- 숫자 X - 숫자 Y: matrix Pearson/Spearman screening
- 범주 X - 숫자 Y: one-hot eta squared screening
- 숫자 X - 범주 Y: one-hot eta squared / point-biserial detailed analysis
- 범주 X - 범주 Y: one-hot contingency / Cramer's V screening

범주형은 label encoding하지 않고 one-hot matrix를 사용합니다.

## 저장 산출물

사용자에게 보이는 최종 저장 산출물은 두 종류만 제공합니다.

1. `output/YYYYMMDD_HHMMSS_run/graphs/*.png`
2. `output/YYYYMMDD_HHMMSS_run/captures/significance_analysis_page.png`

Excel/HTML/PDF/PPT 리포트, CSV/Parquet 다운로드 버튼은 만들지 않습니다.

내부 처리를 위한 candidate/cache/log 파일은 `output/cache/`, `output/*_run/temp/`, `output/*_run/logs/` 아래에 생성될 수 있습니다. 이 파일들은 사용자-facing 산출물이 아닙니다.

## 소스 배포 주의

`output` 폴더는 실행 중 자동 생성됩니다. 소스 zip에는 `output/cache/`와 과거 `*_run` 결과를 포함하지 않습니다.

배포 제외 대상은 `.gitignore`에 정리되어 있습니다.

- `output/cache/`
- `output/*_run/`
- `__pycache__/`
- `*.pyc`
- `.pytest_cache/`
- `temp/`
- `logs/`

## 대용량 UI 원칙

- 20000개 컬럼을 선택 위젯에 직접 넣지 않습니다.
- 원본 데이터 preview는 head(100), 최대 80컬럼으로 제한합니다.
- 결과 테이블은 Top N만 표시합니다.
- 그래프는 사용자가 선택했을 때만 렌더링합니다.
- 결과 화면은 기본 진입 시 merged parquet를 읽지 않고, 그래프 표시/저장 시점에 필요한 컬럼만 읽습니다.
