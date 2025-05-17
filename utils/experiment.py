import os
import time
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
import requests

from config import ExperimentConfig
from prompts.templates import TEMPLATES
from utils.metrics import evaluate_correction
import concurrent.futures

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        self.template = TEMPLATES[config.template_name]
        self.api_url = config.api_url
        self.model = config.model
    
    def _make_messages(self, text: str) -> List[Dict[str, str]]:
        return [    
            # system
            {"role": "system", "content": self.template},

            # few-shot examples
            # 띄어쓰기, 철자, 문장 부호 오류
            {"role": "user", "content": "뭐든과하지만 안으면조은것같타요"}, 
            {"role": "assistant", "content": "뭐든 과하지만 않으면 좋은 것 같아요."},

            # 띄어쓰기, 어미 오류
            {"role": "user", "content": "증가와 감소 구간을정하는건데요기서 궁금한점이있씁니다."}, 
            {"role": "assistant", "content": "증가와 감소 구간을 정하는 건데 여기서 궁금한 점이 있습니다."},
            
            # 철자, 조사, 띄어쓰기, 문장 부호 오류
            {"role": "user", "content": "전재가실제로 참이라는 근거가지문과보기어디예예도나타나지않았기때문입니다."}, 
            {"role": "assistant", "content": "전제가 실제로 참이라는 근거가 지문과 보기 어디에도 나타나지 않았기 때문입니다."},

            # 조사, 어미, 띄어쓰기 오류
            {"role": "user", "content": "저더비대면이라가지를 못 하구잇네요"}, 
            {"role": "assistant", "content": "저도 비대면이라 가지를 못 하고 있네요."},

            # 띄어쓰기, 철자 오류
            {"role": "user", "content": "거기에 내공문서 윗조해서도도보내더라."}, 
            {"role": "assistant", "content": "거기에 내 공문서 위조해서도 보내더라."},

            # 띄어쓰기, 철자, 어미, 조사 오류
            {"role": "user", "content": "풀이과정깔 끔해 질거에오"}, 
            {"role": "assistant", "content": "풀이 과정이 깔끔해질 거예요."},

            # 띄어쓰기, 철자, 어미, 문장 부호 오류
            {"role": "user", "content": "그래서 주변에 필요한사람들있으면해주려고. 통합과학 기반으로해서 심화적으로 다루듯 공부햇어요! 논술전형에대한유익한정보들 많이올려주시면 감사하게슴니다 내일 밤에 끝낸거 인증 안하면 욕해주세요 "}, 
            {"role": "assistant", "content": "그래서 주변에 필요한 사람들 있으면 해주려고. 통합과학 기반으로 해서 심화적으로 다루듯 공부했어요! 논술전형에 대한 유익한 정보들 많이 올려주시면 감사하겠습니다. 내일 밤에 끝낸 거 인증 안 하면 욕해주세요. "},
            
            # 맞춤법, 띄어쓰기 오류 
            {"role": "user", "content": "이문장이무엇을 이미하는지 잘모르겠습니다. 언래라면 확인하구 너머가야하는것이죠? 예비시행 문제 보니까 저희떄랑은 번호가 좀 바뀐거 같더라구요."}, 
            {"role": "assistant", "content": "이 문장이 무엇을 의미하는지 잘 모르겠습니다. 원래라면 확인하고 넘어가야 하는 것이죠?\
            예비시행 문제 보니까 저희 때랑은 번호가 좀 바뀐 거 같더라고요."},
            
            # 띄어쓰기, 철자, 조사, 어미 오류
            {"role": "user", "content": "내가 풀땐 나 말곤 아무도 못알아보는데 글씨 대박이시네요. 학교가 내신챙기기 쉬운 곳이라 정시공부 어떻게 해야할지 고민 중이였거든요."}, 
            {"role": "assistant", "content": "내가 풀 땐 나 말곤 아무도 못 알아보는데 글씨가 대박이시네요. 학교가 내신 챙기기 쉬운 곳이라 정시 공부를 어떻게 해야 할지 고민 중이었거든요."},

            # 철자, 띄어쓰기 오류
            {"role": "user", "content": "토론은 지두 교수님이 별 말 안헸다."}, 
            {"role": "assistant", "content": "토론은 지도 교수님이 별말 안 했다."},

            # 철자, 띄어쓰기 오류
            {"role": "user", "content": "담단원으로 너머ㅓ가야겟어요..."},
            {"role": "assistant", "content": "다음 단원으로 넘어가야겠어요..."},

            # 띄어쓰기, 철자, 어미, 문장 부호 오류
            {"role": "user", "content": "벼랑끝에 서 있따는 생각으루 해야 한다구 하시더랔"}, 
            {"role": "assistant", "content": "벼랑 끝에 서 있다는 생각으로 해야 한다고 하시더라."},

            # 조사, 철자, 띄어쓰기, 문장 부호
            {"role": "user", "content": "나혼자선줄알았는데 알고보니 다들오고있던거엿어요"}, 
            {"role": "assistant", "content": "나 혼자선 줄 알았는데, 알고 보니 다들 오고 있던 거였어요."},

            {"role": "user", "content": text}
        ]
    
    def _call_api_single(self, text_to_correct: str) -> str: 
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        time.sleep(0.5)
        messages = self._make_messages(text_to_correct)

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p
        }

        ## 429 Error Handling
        max_retries = 10
        for attempt in range(max_retries):
            response = requests.post(self.api_url, headers=headers, json=data)
            
            if response.status_code == 429:
                wait = 2 ** attempt
                print(f"[WARN] 429 Too Many Requests. Retrying in {wait}s...")
                time.sleep(wait)
                continue  # 재시도
            elif not response.ok:
                print(f"[ERROR] HTTP {response.status_code}: {response.text}")
                return f"[ERROR] HTTP {response.status_code}"
            
            try:
                result = response.json()
                output = result["choices"][0]["message"]["content"]
                return output
            except (KeyError, IndexError, ValueError) as e:
                print(f"[ERROR] Unexpected API response format: {result}")
                return "[ERROR] Invalid API response format"

        return "[ERROR] Failed after retries"


    def run_parallel(self, data: pd.DataFrame, max_workers: int = 3) -> pd.DataFrame:

        def process_row(idx, row) -> Dict:
            try:
                text = row['err_sentence']
                if pd.isna(text) or not text.strip():
                    corrected = ''
                else:
                    corrected = self._call_api_single(text)
                return {'idx': idx, 'id': row['id'], 'cor_sentence': corrected}
            except Exception as e:
                return {'idx': idx, 'id': row['id'], 'cor_sentence': f"[ERROR] {str(e)}"}

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_row, idx, row): idx
                for idx, (_, row) in enumerate(data.iterrows())
            }

            try:
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    results.append(future.result())
            except KeyboardInterrupt:
                print("\n[사용자 인터럽트 감지: 실행 중단]")
                executor.shutdown(wait=False, cancel_futures=True)
                raise  

        # 순서 복원
        results_sorted = sorted(results, key=lambda x: x['idx'])
        return pd.DataFrame(results_sorted)[['id', 'cor_sentence']]


    def run_template_experiment(self, train_data: pd.DataFrame, valid_data: pd.DataFrame) -> Dict:
        """템플릿별 실험 실행"""
        print(f"\n=== {self.config.template_name} 템플릿 실험 ===")
        
        # 학습 데이터로 실험
        print("\n[학습 데이터 실험]")
        train_results = self.run_parallel(train_data)
        train_recall = evaluate_correction(train_data, train_results)
        
        # 검증 데이터로 실험
        print("\n[검증 데이터 실험]")
        valid_results = self.run_parallel(valid_data)
        valid_recall = evaluate_correction(valid_data, valid_results)
        
        return {
            'train_recall': train_recall,
            'valid_recall': valid_recall,
            'train_results': train_results,
            'valid_results': valid_results
        } 