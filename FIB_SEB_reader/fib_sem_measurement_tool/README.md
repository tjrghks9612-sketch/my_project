# FIB-SEM Measurement Tool

Python/customtkinter 기반 FIB-SEM ROI 측정 데스크톱 MVP입니다.

## 실행

```powershell
python -m pip install -r fib_sem_measurement_tool\requirements.txt
python -m fib_sem_measurement_tool.main
```

또는:

```powershell
python fib_sem_measurement_tool\main.py
```

## 현재 구현 범위

- 이미지/폴더 불러오기
- 3분할 UI: 메인 이미지, 썸네일/결과 목록, 옵션 패널
- ROI 드래그 지정
- 수동 캘리브레이션 선 드래그
- scale bar line 자동 검출과 실제 길이 입력 기반 보정
- 이미지별 독립 설정
- 그룹 생성/해제 및 그룹 공유 설정
- 설정 우선순위: image_specific > group_shared > global_default
- ROI absolute_copy / relative_copy 적용
- 가로 CD, 세로 THK, 가로+세로 동시 측정
- 한쪽/양쪽 taper 측정
- 빠른 grayscale 변화량 기반 측정
- 최소 옵션 UI: 측정 타입, 대표값, ROI 복사, 캘리브레이션, 실행 범위
- 가벼운 multi-line scan, 이상치 제거, confidence/status 계산
- 현재 이미지 ROI가 있을 때 옵션 변경 후 자동 재측정/overlay 갱신
- 프로필 그래프/overlay/썸네일 갱신 경량화
- overlay 미리보기 및 선택적 overlay 이미지 저장
- 메인 이미지 hover 위치의 가로/세로 grayscale profile graph
- 하단 장비 메타 strip 제거 및 grayscale profile graph 확대
- ROI 드래그 시 체크된 선택 이미지 전체에 즉시 ROI 전파
- utf-8-sig CSV export

## 사용 흐름

1. 이미지를 불러옵니다.
2. 메인 이미지에서 ROI를 드래그합니다.
3. 오른쪽 옵션에서 측정 타입, 경계 기준, 대표값, 노이즈 프리셋을 선택합니다.
4. 필요하면 스케일바 자동 검출 또는 수동 선으로 캘리브레이션을 적용합니다.
5. 현재/선택/그룹/전체 범위로 측정합니다.
6. 결과와 설정 출처를 중앙 목록에서 검토하고 CSV로 저장합니다.
