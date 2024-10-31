input_path="./data/final_data.jsonl"
config_dir="./configs/gpt_4o"
output_dir="./output/gpt_4o"
prompt_template_path="./configs/prompt_template.json"
max_round=15
model="gpt-4o-2024-05-13"
base_url="YOUR_BASE_URL"
api_key="YOUR_API_KEY"
temperature=1
max_tokens=128
judge_config="./configs/judge_config.json"
task_workers=4

python run_eval.py \
	 --input_path $input_path \
     --config_dir $config_dir \
     --output_dir $output_dir \
     --prompt_template_path $prompt_template_path \
     --max_round $max_round \
     --model $model \
     --base_url $base_url \
     --api_key $api_key \
     --temperature $temperature \
     --max_tokens $max_tokens \
     --judge_config $judge_config \
     --task_workers $task_workers
