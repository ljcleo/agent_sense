"""
Batch Evaluation
"""

import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from utils.data_utils import generate_batch_config, generate_heter_batch_config, calculate_tmpl_res
from utils.logger import setup_logger
from simulation import Simulation
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_task(config_file, args):
    task = Simulation.from_task(os.path.join(args.config_dir, config_file), args.output_dir)
    scene_id = task.scene['scene_id']
    
    output_path = os.path.join(args.output_dir, f'{scene_id}.json')
    
    if os.path.exists(output_path):
        res = json.load(open(output_path, 'r'))
    else:
        # simulation and evaluation
        res = task.run()
    
    return scene_id, res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', type=str, default='homo', help='interaction between homogeneous or heterogeneous agents, choose from [homo, heter]')
    parser.add_argument('--input_path', type=str, default="./data/final_data.jsonl", help='path of the file of scenario setting')
    parser.add_argument('--config_dir', type=str, default="./configs/llama2_13b/", help='dir of generated cofig')
    parser.add_argument('--output_dir', type=str, default="./output/llama2_13b/", help='dir of output')
    parser.add_argument('--prompt_template_path', type=str, default="/remote-home/share/xymou_share/socialbench/SENSE/configs/prompt_template_hide.json", help='the path of prompt template for social agents and judge agents')
    parser.add_argument('--max_round', type=int, default=15, help='the max round of dialog, if set to 0, use automatic 10*len(agents)')
    parser.add_argument('--speaker_selection_method', type=str, default='random', help='selection method of slecting the next speaker')
    parser.add_argument('--allow_repeat_speaker', type=bool, default=False, help='whether repeated speakers are allowed. must set False when only 2 agents are involved')
    parser.add_argument('--model', type=str, default="Llama-2-13b-chat-hf", help='the driven models of agents')
    parser.add_argument('--base_url', type=str, default="http://0.0.0.0:8000/v1", help='url of API interface')
    parser.add_argument('--api_key', type=str, default="1234", help='key of API interface')
    parser.add_argument('--api_type', type=str, default="openai", help='type of API interface')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=128)
    parser.add_argument('--judge_config', type=str, default="./configs/judge_config.json", help='the paths of judge configs')    
    parser.add_argument('--option_mark', type=str, default='upper', help='option mark for multiple-choice questions in info reasoning evaluation')   
    parser.add_argument('--task_workers', type=int, default=4, help='num of parallel workers for simulation')

    args = parser.parse_args()

    global logger
    logger = setup_logger('Evaluation', args.output_dir, 0)
    if args.pattern =='homo':
        logger.info('Evaluating model: {}'.format(args.model))
    # generate configs for simulation
    if not os.path.exists(args.prompt_template_path):
        raise Exception("Please provide json of prompt templates.")
    prompt_template = json.load(open(args.prompt_template_path,'r'))
    if args.pattern =='homo':
        generate_batch_config(
            input_path=args.input_path,
            output_dir=args.config_dir,
            prompt_template=prompt_template["prompt_template"],
            judge_prompt_template=prompt_template["judge_prompt_template"],
            max_round=args.max_round,
            speaker_selection_method=args.speaker_selection_method,
            allow_repeat_speaker=args.allow_repeat_speaker,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            api_type=args.api_type,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            judge_config=args.judge_config,
            option_mark=args.option_mark
        )    
    elif args.pattern =='heter':
        generate_heter_batch_config(
            input_path=args.input_path,
            output_dir=args.config_dir,
            prompt_template=prompt_template["prompt_template"],
            judge_prompt_template=prompt_template["judge_prompt_template"],
            max_round=args.max_round,
            speaker_selection_method=args.speaker_selection_method,
            allow_repeat_speaker=args.allow_repeat_speaker,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            judge_config=args.judge_config,
            option_mark=args.option_mark
        )     
    else:
        raise NotImplementedError("unsupported pattern: {}".format(args.pattern))    

    # simulation
    all_res = {}
    configs = [f for f in os.listdir(args.config_dir) if os.path.isfile(os.path.join(args.config_dir, f))]
    logger.info("***** Runing Simulation and Evaluation *****")
    logger.info("  Num tasks = %d", len(configs))

    with ThreadPoolExecutor(max_workers=args.task_workers) as executor:
        futures = {executor.submit(process_task, config_file, args): config_file for config_file in configs}

        for future in tqdm(as_completed(futures), total=len(configs), desc="simulating"):
            config_file = futures[future]
            try:
                scene_id, res = future.result()
                all_res[scene_id] = res
                if res["info_metrics"]:
                    logger.info('Scene {} | goal-self: {} goal-others: {} goal-judge: {} | info: {}'.format(scene_id, round(res["goal_metrics"]["self"],4), 
                                round(res["goal_metrics"]["others"],4), {judge_name: round(res["goal_metrics"][judge_name],4) for judge_name in res["goal_metrics"] if judge_name.startswith("judge")}, 
                                round(res["info_metrics"]["avg"],4)))
                else:
                    logger.info('Scene {} | goal-self: {} goal-others: {} goal-judge: {} | info: {}'.format(scene_id, round(res["goal_metrics"]["self"],4), 
                                round(res["goal_metrics"]["others"],4), {judge_name: round(res["goal_metrics"][judge_name],4) for judge_name in res["goal_metrics"] if judge_name.startswith("judge")}, 
                                "NONE"))
            except Exception as e:
                logger.error(f"Error in simulating {config_file}: {e}")

    # average the results of all the scenarios
    goal_self_score = [all_res[key]['goal_metrics']['self'] for key in all_res]
    goal_others_score = [all_res[key]['goal_metrics']['others'] for key in all_res]
    judges = [name for name in all_res[list(all_res.keys())[0]]['goal_metrics'] if name.startswith('judge')]
    judge_score_dict = {}
    for judge in judges:
        goal_judge_score = [all_res[key]['goal_metrics'][judge] for key in all_res]
        goal_judge_score = np.mean(goal_judge_score)
        judge_score_dict[judge] = round(goal_judge_score,4)
    info_score = [all_res[key]['info_metrics']['avg'] for key in all_res if all_res[key]['info_metrics']]
    goal_self_score = np.mean(goal_self_score)
    goal_others_score = np.mean(goal_others_score)
    info_score = np.mean(info_score)
    
    logger.info('===== Results of Scenarios =====')
    logger.info('the average result of goal completion at self dim: {}'.format(round(goal_self_score,4)))
    logger.info('the average result of goal completion at others dim: {}'.format(round(goal_others_score,4)))
    logger.info('the average result of goal completion at judge dim: {}'.format(judge_score_dict))
    logger.info('the average result of info reasoning: {}'.format(round(info_score,4)))
    
    logger.info('===== Results of Templates =====')
    res = calculate_tmpl_res(all_res, args.input_path)
    logger.info('the average result of goal completion at self dim: mean: {} std: {}'.format(res['goal_self_score'],res['goal_self_std']))
    logger.info('the average result of goal completion at others dim: mean: {} std: {}'.format(res['goal_others_score'],res['goal_others_std']))
    logger.info('the average result of goal completion at judge dim: mean: {} std: {}'.format(res['goal_judge_score'], res['goal_judge_std']))   
    logger.info('the average result of info reasoning: mean: {} std: {}'.format(res['info_score'], res['info_score_std']))
    
    

if __name__=="__main__":
    main()
