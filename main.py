import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from config import ExperimentConfig
from prompts.templates import TEMPLATES
from utils.experiment import ExperimentRunner

def main():
    # API 키 로드
    load_dotenv()
    api_key = os.getenv('UPSTAGE_API_KEY')
    if not api_key:
        raise ValueError("API key not found in environment variables")
    
    # 기본 설정 생성
    base_config = ExperimentConfig(template_name='basic')
    
    # 데이터 로드
    train = pd.read_csv(os.path.join(base_config.data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(base_config.data_dir, 'test.csv'))
    
    # 토이 데이터셋 생성
    toy_data = train.sample(n=base_config.toy_size, random_state=base_config.random_seed).reset_index(drop=True)
        
    # train/valid 분할
    train_data, valid_data = train_test_split(
        toy_data,
        test_size=base_config.test_size,
        random_state=base_config.random_seed
    )
    
    # 모든 템플릿으로 실험
    # 실험 결과를 저장할 딕셔너리
    results = {}

    # 모든 템플릿으로 실험
    for template_name in TEMPLATES.keys():
        config = ExperimentConfig(
            template_name=template_name,
            temperature=base_config.temperature,
            experiment_name=f"toy_experiment_{template_name}"
        )
        runner = ExperimentRunner(config, api_key)
        result = runner.run_template_experiment(train_data, valid_data)
        results[template_name] = result

        # 예측된 교정 문장 리스트 꺼내기
        train_preds = result['train_results']['cor_sentence']
        valid_preds = result['valid_results']['cor_sentence']

        # CSV 저장용 DataFrame 생성
        df_train = pd.DataFrame({
            'err_sentence': train_data['err_sentence'].tolist(),
            'pred_sentence': train_preds,
            'cor_sentence': train_data['cor_sentence'].tolist()
        })
        df_valid = pd.DataFrame({
            'err_sentence': valid_data['err_sentence'].tolist(),
            'pred_sentence': valid_preds,
            'cor_sentence': valid_data['cor_sentence'].tolist()
        })

        # 오류가 있는 행만 필터링
        df_train_errors = df_train[df_train['pred_sentence'] != df_train['cor_sentence']]
        df_valid_errors = df_valid[df_valid['pred_sentence'] != df_valid['cor_sentence']]

        # 폴더 이름 변경
        train_dir = os.path.join(template_name, "train_errors")
        valid_dir = os.path.join(template_name, "val_errors")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)

        # 파일 이름 변경
        train_csv_name = f"{template_name}_train_errors.csv"
        valid_csv_name = f"{template_name}_val_errors.csv"
        train_csv_path = os.path.join(train_dir, train_csv_name)
        valid_csv_path = os.path.join(valid_dir, valid_csv_name)

        # 오류가 있을 때만 저장 및 출력
        if not df_train_errors.empty:
            df_train_errors.to_csv(train_csv_path, index=False, encoding='utf-8-sig')
            print(f"[{template_name}] train errors saved → {train_csv_path}")
        if not df_valid_errors.empty:
            df_valid_errors.to_csv(valid_csv_path, index=False, encoding='utf-8-sig')
            print(f"[{template_name}] valid errors saved → {valid_csv_path}")

    # 결과 비교 출력
    print("\n=== 템플릿별 성능 비교 ===")
    # 검증 셋 비율 (train/valid 분할 시 test_size)
    val_ratio = base_config.test_size 
    train_ratio = 1 - val_ratio

    for template_name, result in results.items():
        tr = result['train_recall']
        vr = result['valid_recall']
        print(f"\n[{template_name} 템플릿]")
        print(f"  Train Recall:    {tr['recall']:.2f}%")
        print(f"  Train Precision: {tr['precision']:.2f}%")
        print(f"  Valid Recall:    {vr['recall']:.2f}%")
        print(f"  Valid Precision: {vr['precision']:.2f}%")
        # 가중평균 리콜 계산 및 출력
        weighted_recall = tr['recall'] * train_ratio + vr['recall'] * val_ratio
        print(f"  Weighted Recall (train {train_ratio:.2f} / val {val_ratio:.2f}): {weighted_recall:.2f}%")
    
    # 최고 성능 템플릿 찾기
    best_template = max(
        results.items(), 
        key=lambda x: x[1]['valid_recall']['recall']
    )[0]
    
    print(f"\n최고 성능 템플릿: {best_template}")
    print(f"Valid Recall: {results[best_template]['valid_recall']['recall']:.2f}%")
    print(f"Valid Precision: {results[best_template]['valid_recall']['precision']:.2f}%")
    
    # 최고 성능 템플릿으로 제출 파일 생성
    print("\n=== 테스트 데이터 예측 시작 ===")
    print("\n=== 부분 테스트 데이터 예측 시작 ===")
    # 1) 공통으로 쓸 설정과 실행기 생성
    config = ExperimentConfig(
        template_name=best_template,
        temperature=base_config.temperature,
        experiment_name="final_submission"
    )
    runner = ExperimentRunner(config, api_key)

    # 2) 처음 100개에 대해서만 중간 저장
    partial_df = test.iloc[:100].reset_index(drop=True)
    partial_results = runner.run_parallel(partial_df)
    pd.DataFrame({
        'id': partial_df['id'],
        'cor_sentence': partial_results['cor_sentence']
    }).to_csv("submission_partial_100.csv", index=False, encoding='utf-8-sig')
    print("처음 100개 중간 파일 저장: submission_partial_100.csv")

    # 3) 전체 데이터 예측 및 최종 저장
    print("\n=== 전체 테스트 데이터 예측 시작 ===")
    full_results = runner.run_parallel(test)
    output = pd.DataFrame({
        'id': test['id'],
        'cor_sentence': full_results['cor_sentence']
    })
    output.to_csv("submission_baseline.csv", index=False, encoding='utf-8-sig')
    print("\n최종 제출 파일 생성: submission_baseline.csv")
    print(f"사용된 템플릿: {best_template}")
    print(f"예측된 샘플 수: {len(output)}")

if __name__ == "__main__":
    main()