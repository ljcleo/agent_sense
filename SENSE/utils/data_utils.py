import random
import yaml
import json
import os
import collections
import numpy as np

def question_with_options(item, option_mark='random'):
    alphabet = ['abcdefghijklmnopqrstuvwxyz',
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            '123456789']
    ret = item['question']
    if option_mark == 'number':
        ab = alphabet[2]
    elif option_mark == 'lower':
        ab = alphabet[0]
    elif option_mark == 'upper':
        ab = alphabet[1]
    else:
        ab = alphabet
    if 'options' in item:
        ret += ' Options: '
        if isinstance(ab, list):
            current_ab = random.choice(ab)
        else:
            current_ab = ab
        for i, opt in enumerate(item['options']):
            ret += '({}) {}'.format(current_ab[i], opt)
            if i == len(item['options']) - 1:
                ret += '.' # + seps[0]
            else:
                ret += '; '
    instruct = "Please answer the question and only output your choice.\n"
    return instruct+ret


def generate_batch_config(
    input_path,
    output_dir,
    prompt_template,
    judge_prompt_template,
    max_round,
    speaker_selection_method,
    allow_repeat_speaker,
    model,
    base_url,
    api_key,
    api_type,
    temperature,
    max_tokens,
    judge_config,
    option_mark
    ):
    lines = open(input_path,'r').readlines()
    judge_config = json.load(open(judge_config,'r'))
    for line in lines:
        line = json.loads(line)

        if max_round<1:
            # using automatic round settings
            max_round = len(line["characters"])*10

        scene={
            "scene_id":line["sample_idx"],
            "background":line["background"],
            "desc":line["description"],
            "goal_question":{char["name"]:char["goals"] for char in line["characters"]},
            "info_question":{char["name"]:char["info_reason_questions"] for char in line["characters"]},
            "option_mark":option_mark,
        }

        groupchat={
            "messages":[],
            "max_round":max_round,
            "speaker_selection_method":speaker_selection_method,
            "allow_repeat_speaker":allow_repeat_speaker,
        }
        
        agents=[
            {
                "name":agent["name"],
                "profile":agent["profile"],
                "social_goal":[d["goal"] for d in agent["goals"]],
                "private_info":agent["private_info"] if agent["private_info"] else "",
                "prompt_template": prompt_template,
                "llm":{
                    "model":model,
                    "base_url":base_url,
                    "api_key":api_key,
                    "api_type":api_type,
                    "temperature":temperature,
                    "max_tokens":max_tokens
                }

            }
            for agent in line["characters"]
        ]

        judge_agents=[
            {
            "name":"judge_"+config['judge_model'],
            "prompt_template":judge_prompt_template,
            "llm":{
                "model":config['judge_model'],
                "base_url":config['judge_base_url'],
                "api_key":config['judge_api_key'],
                "api_type":config['judge_api_type'],
                "temperature":config['judge_temperature'],
                "max_tokens":config['judge_max_tokens']
                }              
            }
            for config in judge_config
        ]
        
        data={
            "scene":scene,
            "groupchat":groupchat,
            "agents":agents,
            "judge_agents":judge_agents
        }

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_dir+f'/{line["sample_idx"]}.yaml', 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def generate_heter_batch_config(
    input_path,
    output_dir,
    prompt_template,
    judge_prompt_template,
    max_round,
    speaker_selection_method,
    allow_repeat_speaker,
    temperature,
    max_tokens,
    judge_config,
    option_mark
    ):
    """using different llms for agents"""
    lines = open(input_path,'r').readlines()
    judge_config = json.load(open(judge_config,'r'))
    for line in lines:
        line = json.loads(line)

        if max_round<1:
            # using automatic round settings
            max_round = len(line["characters"])*10

        scene={
            "scene_id":line["sample_idx"],
            "background":line["background"],
            "desc":line["description"],
            "goal_question":{char["name"]:char["goals"] for char in line["characters"]},
            "info_question":{char["name"]:char["info_reason_questions"] for char in line["characters"]},
            "option_mark":option_mark,
        }

        groupchat={
            "messages":[],
            "max_round":max_round,
            "speaker_selection_method":speaker_selection_method,
            "allow_repeat_speaker":allow_repeat_speaker,
        }
        agents=[
            {
                "name":agent["name"],
                "profile":agent["profile"],
                "social_goal":[d["goal"] for d in agent["goals"]],
                "private_info":agent["private_info"] if agent["private_info"] else "",
                "prompt_template": prompt_template,
                "llm":{
                    "model":agent["model"],
                    "base_url":agent["base_url"],
                    "api_key":agent["api_key"],
                    "api_type":agent["api_type"],
                    "temperature":temperature,
                    "max_tokens":max_tokens
                }

            }
            for agent in line["characters"]
        ]

        judge_agents=[
            {
            "name":"judge_"+config['judge_model'],
            "prompt_template":judge_prompt_template,
            "llm":{
                "model":config['judge_model'],
                "base_url":config['judge_base_url'],
                "api_key":config['judge_api_key'],
                "api_type":config['judge_api_type'],
                "temperature":config['judge_temperature'],
                "max_tokens":config['judge_max_tokens']
                }              
            }
            for config in judge_config
        ]
        
        data={
            "scene":scene,
            "groupchat":groupchat,
            "agents":agents,
            "judge_agents":judge_agents
        }

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_dir+f'/{line["sample_idx"]}.yaml', 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def calculate_tmpl_res(all_res, input_path):
    data = open(input_path, 'r').readlines()
    data = [json.loads(s) for s in data]
    template_dict = collections.defaultdict(list)
    for d in data:
        template_dict[d['template_idx']].append(d['sample_idx'])
    template_res = {}
    judges = [name for name in all_res[list(all_res.keys())[0]]['goal_metrics'] if name.startswith('judge')]
    for template_idx in template_dict:
        goal_self_score = [all_res[key]['goal_metrics']['self'] for key in template_dict[template_idx]]
        goal_others_score = [all_res[key]['goal_metrics']['others'] for key in template_dict[template_idx]]
        judge_score_dict = {}
        judge_std_dict = {}
        for judge in judges:
            goal_judge_score = [all_res[key]['goal_metrics'][judge] for key in template_dict[template_idx]]
            goal_judge_std = np.std(goal_judge_score,ddof=1)
            goal_judge_score = np.mean(goal_judge_score)
            judge_score_dict[judge] = goal_judge_score
            judge_std_dict[judge] = goal_judge_std
        info_score = [all_res[key]['info_metrics']['avg'] for key in template_dict[template_idx] if all_res[key]['info_metrics']]
        goal_self_std = np.std(goal_self_score,ddof=1)
        goal_self_score = np.mean(goal_self_score)
        goal_others_std = np.std(goal_others_score,ddof=1)
        goal_others_score = np.mean(goal_others_score)
        info_score_std = np.std(info_score,ddof=1) if len(info_score)>1 else 'NONE'    
        info_score = np.mean(info_score) if len(info_score) else 'NONE'     
        template_res[template_idx] = {
            'goal_self_score':goal_self_score,
            'goal_self_std':goal_self_std,
            'goal_others_score':goal_others_score,
            'goal_others_std':goal_others_std,
            'goal_judge_score':judge_score_dict,
            'goal_judge_std':judge_std_dict,            
            'info_score':info_score,
            'info_score_std':info_score_std
            }
    print('# of templates:', len(template_res))
    goal_self_score = np.mean([template_res[key]['goal_self_score'] for key in template_res])
    goal_self_std = np.mean([template_res[key]['goal_self_std'] for key in template_res])
    goal_others_score = np.mean([template_res[key]['goal_others_score'] for key in template_res])
    goal_others_std = np.mean([template_res[key]['goal_others_std'] for key in template_res])  
    goal_judge_score, goal_judge_std = {},{}
    for judge in judges:
        goal_judge_score[judge] = round(np.mean([template_res[key]['goal_judge_score'][judge] for key in template_res]), 4) 
        goal_judge_std[judge] = round(np.mean([template_res[key]['goal_judge_std'][judge] for key in template_res]), 4)
    info_score = np.mean([template_res[key]['info_score'] for key in template_res if type(template_res[key]['info_score'])!=str])
    info_score_std = np.mean([template_res[key]['info_score_std'] for key in template_res if type(template_res[key]['info_score_std'])!=str])        
    res = {
            'goal_self_score':round(goal_self_score,4),
            'goal_self_std':round(goal_self_std,4),
            'goal_others_score':round(goal_others_score,4),
            'goal_others_std':round(goal_others_std,4),
            'goal_judge_score':goal_judge_score,
            'goal_judge_std':goal_judge_std,          
            'info_score':round(info_score,4),
            'info_score_std':round(info_score_std, 4)
            }
    return res


if __name__ == "__main__":
    pass
