input_path="./data/final_data.jsonl"
config_dir="./configs/llama3_8b"
output_dir="./output/llama3_8b"
prompt_template_path="./configs/prompt_template.json"
max_round=15
model="llama-3-8b"
base_url="LOCAL_SERVER_URL"
api_key="1234"
api_type="openai"
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
     --api_type $api_type \
     --temperature $temperature \
     --max_tokens $max_tokens \
     --judge_config $judge_config \
     --task_workers $task_workers
