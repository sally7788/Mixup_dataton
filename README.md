# 🧪 MixUp_오리훈제구이 : Grammar Error Correction Promptathon 

본 레포지토리는 Grammar Error Correction Promptathon  실험을 재현하고 확장하기 위한 코드 및 가이드를 제공합니다.

**단, 반드시 평가 코드는 제외후 제출합니다**

## 📌 프로젝트 개요

* **목표**: ex. Solar Pro API를 활용하여 프롬프트 만으로 한국어 맞춤법 교정 성능을 개선한다. 
* **접근 전략**:

---

* **주요 실험 내용**:
| 접근 방식              | 실험 내용 | 한계 및 문제점 |
|------------------------|-----------|----------------|
| **Zero-shot**          | - 단순 지시문 기반 교정<br>- "띄어쓰기와 맞춤법을 고쳐줘" 형태의 프롬프트 사용 | - 구어체·조사 생략 등 비형식적 문장에 취약<br>- 오류 유형에 대한 일반화 부족 |
| **Rule-based**         | - 형태소 분석 기반 규칙 적용<br>- 띄어쓰기/조사/의존 명사 보정 룰 구성 | - 의미 단위와 충돌 시 부자연스러운 결과<br>- 관용 표현, 문맥 해석에 한계 |
| **CoT & Feedback Loop** | - 한 문장을 여러 턴에 나눠 교정<br>- 사용자 피드백 기반 반복 보정 시도 | - 문맥 유지가 어려워 응답 일관성 저하<br>- 처리 시간 및 흐름 단절 발생 |
| **Few-shot Prompt**    | - 다양한 오류 유형을 포함한 예시 설계<br>- 문체(반말, 해요체, 합쇼체) 혼용, 복합 오류 상황 포함 | - 예시 설계에 시간과 정제 필요<br>- 출력 문체 일관성 유지에 신경 써야 함 |



---

## ⚙️ 환경 세팅 & 실행 방법

### 1. 사전 준비 

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo/experiment
```

### 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 실험 실행

```bash
python run_experiment.py --input sample_input.txt --output result.json
```

> 📎 실행 옵션 (예시):
> `--input`: 실험 대상 파일
> `--output`: 결과 저장 파일 경로

---


## 🚧 실험의 한계 및 향후 개선

* **한계**:

  * 긴 문장/복문에서 누락되는 오류 존재
  * 도메인 특화 문서(법률/의료 등)에서는 성능 저하
* **향후 개선 방향**:

  * 오류 유형 자동 분류 → 맞춤형 프롬프트 분기
  * User Feedback loop를 통한 교정 정확도 향상

---

## 📂 폴더 구조

```
📁 code/
├── main.py              # 메인 실행 파일
├── config.py            # 설정 파일
├── requirements.txt     # 필요한 패키지 목록
├── __init__.py         # 패키지 초기화 파일
├── utils/              # 유틸리티 함수들
│   ├── __init__.py     # utils 패키지 초기화
│   ├── experiment.py   # 실험 실행 및 API 호출
│   └── metrics.py      # 평가 지표 계산
└── prompts/            # 프롬프트 템플릿 저장
    ├── __init__.py     # prompts 패키지 초기화
    └── templates.py    # 프롬프트 템플릿 정의
```
