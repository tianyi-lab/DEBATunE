
python evaluation/eval_controllability/wrap_eval.py \
    --result_file inference_results_of_exsisting_models/DEBATunE_7b \
    --save_name evaluation/eval_controllability/results/DEBATunE_7b.json \
    --api_key xxx \
    --api_model gpt-4 \
    --batch_size 10

python evaluation/eval_controllability/check.py --result_file evaluation/eval_controllability/results/DEBATunE_7b.json