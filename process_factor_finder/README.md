# Process Factor Finder

제조/공정 X 데이터와 품질/특성 Y 데이터를 Key 기준으로 병합한 뒤, Y와 함께 움직이는 X 후보 인자를 빠르게 찾는 Streamlit 대시보드입니다.

이 도구는 원인을 확정하지 않습니다. p-value만 보지 않고 효과 크기, 모델 중요도, 안정성, 데이터 품질을 함께 보면서 “우선 확인할 유의 후보”를 줄여주는 실무 탐색 도구입니다.

## 1. 주요 기능

- CSV, XLSX, Parquet 파일 업로드
- 여러 X 파일끼리 먼저 Key 병합
- 여러 Y 파일끼리 먼저 Key 병합
- 최종 X/Y Key 병합
- 단일 Y 기준 유의인자 분석
- 전체 X-Y 유의쌍 탐색
- 대용량 wide data용 제한 렌더링, chunk 처리, 중간 저장
- 분석 결과 CSV/Parquet/Excel/HTML 저장
- 상위 결과 그래프 자동 저장

## 2. 설치

```powershell
cd process_factor_finder
python -m pip install -r requirements.txt
```

가상환경을 쓰려면 아래처럼 실행합니다.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 3. 예시 데이터 생성

실제 사내 데이터는 포함하지 않습니다. 먼저 가짜 제조 예시 데이터를 생성합니다.

```powershell
python sample_data_generator.py
```

생성 파일:

- `data/sample_x.csv`
- `data/sample_y.csv`
- `data/sample_x2.csv`
- `data/sample_y2.csv`

대용량 테스트용 wide data가 필요하면 다음을 사용합니다.

```powershell
python wide_data_generator.py --rows 5000 --x-cols 20000 --y-cols 10
```

## 4. 실행

Python 소스 실행:

```powershell
streamlit run app.py
```

exe 실행:

```powershell
dist\ProcessFactorFinder.exe
```

브라우저가 자동으로 열리지 않으면 주소창에 직접 입력합니다.

```text
http://localhost:8501
```

8501 포트가 이미 사용 중이면 앱이 다른 포트를 찾거나, 실행 중인 Streamlit/ProcessFactorFinder 프로세스를 종료한 뒤 다시 실행하세요.

## 5. 기본 사용 순서

1. 왼쪽 사이드바에서 X/Y 파일을 업로드합니다.
2. `병합/프로파일` 화면에서 X 파일 간 Key, Y 파일 간 Key를 선택하고 내부 병합을 실행합니다.
3. 최종 X Key와 Y Key를 선택하고 X/Y 최종 병합을 실행합니다.
4. `유의인자 분석` 화면으로 이동합니다.
5. 단일 Y 분석 또는 전체 X-Y 유의쌍 탐색을 실행합니다.
6. 상위 후보, 상세 설명, 그래프, 다운로드 파일을 확인합니다.

## 6. 설정 방식

대용량 데이터 처리를 기본 동작으로 사용합니다. 별도의 “대용량 모드” 토글은 없습니다.

분석 옵션은 왼쪽 사이드바의 `설정 > 분석 설정 열기` 안에 모았습니다. 평소에는 화면을 가볍게 유지하고, 필요할 때만 설정을 열어 조정합니다.

주요 설정:

- `최대 결측률`: 이 비율보다 결측이 많은 컬럼은 후보에서 제외합니다.
- `최소 조건별 표본수`: 범주형 값, 위치, 조건별 최소 데이터 수입니다.
- `단일 Y Top N`: 단일 Y 분석에서 표시할 후보 개수입니다.
- `Y별 상세 계산 후보 수`: matrix screening 후 각 Y마다 상세 계산할 X 후보 수입니다.
- `전체 상세 계산 후보 상한`: 전체 X-Y 탐색에서 실제 상세 점수까지 계산할 최대 후보 수입니다.
- `Matrix X/Y chunk size`: 행렬 계산을 나누어 처리하는 컬럼 단위입니다.
- `Matrix screening 방식`: Pearson은 빠르고, Spearman은 순위 기반이라 더 느리지만 단조 패턴을 볼 수 있습니다.
- `중간 결과 저장`: 대용량 분석 중 결과와 오류 로그를 파일로 저장합니다.
- `그래프 자동 저장`: 상위 후보 그래프를 자동 저장합니다.

## 7. 단일 Y 기준 분석

Y 하나를 선택하고 해당 Y와 관련이 커 보이는 X 후보를 찾습니다.

이 모드는 numeric, binary, categorical Y를 지원하며 X 타입에 따라 Spearman, Kruskal-Wallis, Mann-Whitney U, Chi-square 등을 사용합니다.

예시:

- Y가 `DPU`이면 `Slot`, `Chamber`가 상위 후보로 나올 수 있습니다.
- Y가 `CD`이면 `Etch_Time`이 상위 후보로 나올 수 있습니다.
- Y가 `Taper`이면 `Anneal_Temp`가 상위 후보로 나올 수 있습니다.
- Y가 `Flicker_NG`이면 `O2_Flow`가 상위 후보로 나올 수 있습니다.

범주형 변수의 해석 문구는 “그룹” 대신 값, 조건, 위치 같은 표현을 사용합니다. 예를 들어 `Slot=1 위치에서 DPU 중앙값이 상대적으로 높게 나타났습니다`처럼 표시됩니다.

## 8. 전체 X-Y 유의쌍 탐색

이 모드는 대용량 wide data를 빠르게 훑기 위해 numeric X와 numeric Y 조합만 행렬 계산으로 분석합니다.

흐름:

1. numeric X 후보와 numeric Y 후보를 고릅니다.
2. 전체 numeric-numeric 조합 수를 표시합니다.
3. NumPy matrix 연산으로 correlation/R²를 계산합니다.
4. Y별 상위 X 후보만 남깁니다.
5. 남은 후보에만 기존 상세 분석과 scoring을 적용합니다.
6. 결과와 그래프를 저장합니다.

문자형/범주형 조합은 이 모드에서 돌지 않습니다. 범주형 조건, Chamber, Recipe, Defect Mode 같은 변수는 단일 Y 기준 분석에서 확인하세요.

이렇게 바꾼 이유는 5000행 x 20000컬럼급 데이터에서 모든 X-Y pair를 Python loop로 훑는 방식이 너무 느리기 때문입니다. 전체 X-Y 모드는 “행렬로 빠르게 후보 압축” 역할에 집중합니다.

## 9. 대용량 데이터 대응

대용량 동작은 기본값입니다.

- 원본 데이터는 `head(100)` 또는 제한된 컬럼만 미리보기로 표시합니다.
- 컬럼 선택은 전체 사용, regex include/exclude, 붙여넣기 방식 중심입니다.
- 20000개 컬럼을 multiselect에 전부 렌더링하지 않습니다.
- 전체 결과 테이블은 화면에 모두 뿌리지 않고 Top N만 표시합니다.
- 전체 결과는 파일로 저장합니다.
- 분석 중 오류가 난 pair는 전체 실행을 멈추지 않고 error log에 기록합니다.
- 분석 화면에서는 무거운 Plotly 그래프를 자동으로 계속 렌더링하지 않습니다. `그래프 표시` 토글을 켠 경우에만 화면에 그립니다.
- 전체 X-Y 분석 후보는 화면 진입 시 전체 컬럼 프로파일링을 하지 않고, dtype 기반 numeric 후보를 빠르게 잡은 뒤 matrix screening 단계에서 품질 필터를 적용합니다.

## 10. 저장 위치

실행 기준 폴더의 `output` 아래에 저장됩니다. exe로 실행하면 exe가 있는 폴더 기준입니다.

주요 폴더:

- `output/results/YYYYMMDD_HHMMSS/`: 분석 결과, 후보, 오류 로그, 제외 컬럼 목록
- `output/plots/YYYYMMDD_HHMMSS/`: 자동 저장 그래프
- `output/reports/`: 리포트 파일
- `output/benchmarks/`: 벤치마크 결과

주요 파일:

- `pairwise_detailed_result.csv`
- `pairwise_detailed_result.parquet`
- `top_results_by_y.csv`
- `x_feature_repeat_summary.csv`
- `matrix_screening_candidates.csv`
- `matrix_screening_candidates.parquet`
- `dropped_columns.csv`
- `screening_log.csv`
- `error_log.csv`
- `plot_index.csv`

## 11. exe 빌드

빌드 도구 설치:

```powershell
python -m pip install -r requirements-build.txt
```

빌드:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

생성 파일:

```text
dist/ProcessFactorFinder.exe
```

## 12. 사내 폐쇄망 실행

방법 A: exe만 전달

1. 인터넷이 되는 PC에서 exe를 빌드합니다.
2. `dist/ProcessFactorFinder.exe`를 폐쇄망 PC로 옮깁니다.
3. 필요하면 exe와 같은 폴더에 `config.yaml`, `data`, `output` 폴더를 둡니다.
4. exe를 실행합니다.
5. 브라우저에서 `http://localhost:8501`을 엽니다.

방법 B: Python 소스로 실행

인터넷 PC에서 wheel 파일을 미리 받습니다.

```powershell
python -m pip download -r requirements.txt -d wheels
```

폐쇄망 PC로 프로젝트 폴더와 `wheels` 폴더를 옮긴 뒤 설치합니다.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --no-index --find-links .\wheels -r requirements.txt
streamlit run app.py
```

## 13. 한계와 주의사항

- 본 결과는 원인 확정이 아니라 우선 확인 후보입니다.
- 전체 X-Y 탐색은 많은 조합을 동시에 보므로 우연히 유의해 보이는 결과가 포함될 수 있습니다.
- 보정 p-value, 효과 크기, 모델 중요도, 안정성, 데이터 품질을 함께 확인하세요.
- Key 병합 품질이 낮으면 분석 결과도 흔들립니다.
- 실제 사내 데이터 적용 전에는 컬럼 정의, 단위, 공정 순서, 설비 이력, 검사 시점을 반드시 확인하세요.
- 보안 데이터, 개인정보, 고객 정보는 샘플이나 리포트에 넣지 마세요.
