
python evaluation/eval_quality/wrap.py \
    --dataset_name debate_test_arg \
    --fname1 inference_results_of_exsisting_models/DEBATunE_7b \
    --fname2 inference_results_of_exsisting_models/vicuna_7b_15 \
    --save_name DEBATunE_7b-VS-vicuna_7b_15 

python eval_quality/wrap_eval.py \
    --wraped_file evaluation/eval_quality/results/DEBATunE_7b-VS-vicuna_7b_15/debate_test_arg_wrap.json \
    --api_key xxx \
    --api_model gpt-4 \
    --batch_size 10

python evaluation/eval_quality/wrap_review_score.py \
    --review_home_path evaluation/eval_quality/results/DEBATunE_7b-VS-vicuna_7b_15/debate_test_arg_wrap_reviews_gpt4.json