# 🧪 MixUp_오리훈제구이 : Grammar Error Correction Promptathon 

본 레포지토리는 Grammar Error Correction Promptathon  실험을 재현하고 확장하기 위한 코드 및 가이드를 제공합니다.

**단, 반드시 평가 코드는 제외후 제출합니다**

## 📌 프로젝트 개요

* **목표**: ex. Solar Pro API를 활용하여 프롬프트 만으로 한국어 맞춤법 교정 성능을 개선한다. 
* **접근 전략**:

  * ex. 오류 유형별 대응 전략 수립 → 반복 실험을 통해 개선
* **주요 실험 내용**:

  * 실험 진행 방식 작성
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
