# Single Format Element Consistency Analysis

이 디렉토리는 하나의 포맷 요소만 다른 포맷 쌍들 간의 일관성(consistency)을 분석하는 코드를 포함합니다.

## 개요

8개의 서로 다른 포맷을 3비트 이진수로 표현하여 분석합니다:
- Format 0: 000, Format 1: 001, Format 2: 010, Format 3: 011
- Format 4: 100, Format 5: 101, Format 6: 110, Format 7: 111

### 분석 쌍 (Analysis Pairs)

각 포맷 요소의 효과를 측정하기 위해 정확히 하나의 비트만 다른 쌍들을 분석합니다:

- **마지막 비트 효과** (Last bit effect): (000,001), (010,011), (100,101), (110,111)
- **가운데 비트 효과** (Middle bit effect): (000,010), (001,011), (100,110), (101,111)  
- **첫번째 비트 효과** (First bit effect): (000,100), (001,101), (010,110), (011,111)

## 파일 구조

```
rebuttal-pair/
├── single_format_element_consistency.py  # 메인 분석 코드
├── single_element_analysis.ipynb        # Jupyter 노트북 (대화형 분석)
├── test_analysis.py                     # 빠른 테스트 스크립트
├── README.md                           # 이 파일
└── results/                            # 분석 결과 출력 디렉토리
```

## 사용 방법

### 1. 빠른 테스트
```bash
cd /data8/baek/RoLM/src/analysis/rebuttal-pair
python test_analysis.py
```

### 2. 전체 분석 실행
```bash
python single_format_element_consistency.py
```

### 3. 대화형 분석 (Jupyter 노트북)
```bash
jupyter notebook single_element_analysis.ipynb
```

## 주요 함수들

### `get_format_pairs_by_position()`
각 비트 위치별로 하나의 비트만 다른 포맷 쌍들을 반환합니다.

### `analyze_single_element_consistency(data_path)`
특정 데이터 파일에 대해 단일 요소 일관성을 분석합니다.

### `analyze_dataset_model_combination(dataset_name, model_name, prompting_strategy)`
특정 데이터셋-모델-전략 조합에 대해 분석을 수행합니다.

### `generate_analysis_report(dataset_names, model_names, prompting_strategies)`
모든 조합에 대한 종합적인 분석 리포트를 생성합니다.

## 입력 데이터 형식

분석 스크립트는 다음 형식의 JSONL 파일을 기대합니다:
```json
{
  "id": "26",
  "predictions": {
    "0": "True",
    "1": "True", 
    "2": "True",
    "3": "True",
    "4": "True",
    "5": "False",
    "6": "True",
    "7": "True"
  },
  "accuracy": {...},
  "consistency": {...}
}
```

## 출력 결과

### 1. 상세 결과 (JSON)
- `results/single_element_consistency_detailed.json`: 모든 조합에 대한 상세 결과

### 2. 요약 결과 (CSV)
- `results/single_element_consistency_summary.csv`: 전체 요약
- `results/first_bit_consistency_summary.csv`: 첫번째 비트 효과
- `results/middle_bit_consistency_summary.csv`: 가운데 비트 효과  
- `results/last_bit_consistency_summary.csv`: 마지막 비트 효과

### 3. 결과 해석

각 비트 위치별 일관성 점수는 다음을 의미합니다:
- **1.0**: 해당 포맷 요소를 바꿔도 예측이 항상 일치
- **0.0**: 해당 포맷 요소를 바꾸면 예측이 항상 불일치
- **0.5**: 해당 포맷 요소를 바꿨을 때 50% 확률로 예측이 일치

## 예제 사용법

```python
from single_format_element_consistency import analyze_dataset_model_combination

# 특정 조합 분석
results = analyze_dataset_model_combination(
    dataset_name="100TFQA",
    model_name="DeepSeek-R1-Distill-Llama-8B", 
    prompting_strategy="few-shot-cot"
)

# 결과 출력
for position, position_results in results.items():
    print(f"{position}: {position_results['mean']:.3f}")
```

## 요구사항

- Python 3.7+
- numpy
- pandas  
- jsonlines
- scipy (통계 분석용)
- matplotlib, seaborn (시각화용, 노트북에서만 필요)

## 결과 분석 가이드

1. **높은 일관성 (>0.8)**: 해당 포맷 요소가 모델 예측에 거의 영향을 주지 않음
2. **중간 일관성 (0.4-0.8)**: 해당 포맷 요소가 모델 예측에 부분적 영향
3. **낮은 일관성 (<0.4)**: 해당 포맷 요소가 모델 예측에 큰 영향을 줌

이 분석을 통해 어떤 포맷 요소가 모델의 일관성에 가장 큰 영향을 미치는지 파악할 수 있습니다.
