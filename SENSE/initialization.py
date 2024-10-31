"""
Initialize from conifg
"""
import os
import yaml
from typing import Dict, List, TYPE_CHECKING
from string import Template
import autogen
from autogen import ConversableAgent, GroupChat
from utils.data_utils import question_with_options

def fill_prompt_template(prompt_template, name, profile, social_goal, private_info, background, desc) -> str:
    """Fill the placeholders in the prompt template"""
    input_arguments = {
        "name": name,
        "profile": profile,
        "social_goal": " ".join(social_goal),
        "private_info": private_info if private_info else 'N/A',
        "background": background,
        "desc": desc,
    }
    return Template(prompt_template).safe_substitute(input_arguments)



def load_scene(scene_config:Dict):
    # prepare questions with options
    info_question = scene_config["info_question"]
    option_mark = scene_config["option_mark"]
    assert option_mark in ["all","upper","lower","number"]
    for agent_name in info_question:
        for i in range(len(info_question[agent_name])):
            question = question_with_options(info_question[agent_name][i], option_mark=option_mark)
            info_question[agent_name][i]['question_with_options'] = question
    scene_config["info_question"] = info_question
    return scene_config


def load_llm_config(llm_config: Dict):
    llm_config = {"config_list": [llm_config], "cache_seed": None}
    return llm_config

def load_agent(agent_config: Dict, scene_config: Dict) -> ConversableAgent:
    agent_config["system_prompt"] = fill_prompt_template(
        agent_config["prompt_template"],
        agent_config["name"], agent_config["profile"], 
        agent_config["social_goal"], agent_config["private_info"],
        scene_config["background"], scene_config["desc"]
    )

    valid_agent_config = {
        "name": agent_config["name"],
        "system_message":agent_config["system_prompt"],
        "llm_config":agent_config["llm_config"],
        "human_input_mode":"NEVER",
    }

    agent = ConversableAgent(**valid_agent_config)
    return agent

def load_judge_agent(agent_config:Dict):
    valid_agent_config = {
        "name": agent_config["name"],
        "system_message":agent_config["prompt_template"],
        "llm_config":agent_config["llm_config"],
        "human_input_mode":"NEVER",
    }

    agent = ConversableAgent(**valid_agent_config)
    return agent    

def load_groupchat(agent_list: List, groupchat_config: Dict):
    group_chat = autogen.GroupChat(
        agent_list,
        **groupchat_config
    ) 
    chat_manager = autogen.GroupChatManager(group_chat)
    return group_chat, chat_manager


def prepare_task_config(config_path):
    """Read the yaml config of the given task in `tasks` directory."""
    if not os.path.exists(config_path):
        raise ValueError(f"Task {config_path} not found.")
    task_config = yaml.safe_load(open(config_path))
    for i, agent_configs in enumerate(task_config["agents"]):
        llm_config = load_llm_config(agent_configs.get("llm", None))
        agent_configs["llm_config"] = llm_config
    for i, agent_configs in enumerate(task_config["judge_agents"]):
        llm_config = load_llm_config(agent_configs.get("llm", None))
        agent_configs["llm_config"] = llm_config
    return task_config


def update_agent_llm_config(agent, key, new_value):
    if key not in agent.llm_config['config_list'][0]:
        raise ValueError(f"key '{key}' is not in llm config of agent {agent.name}.")
    agent.llm_config['config_list'][0][key] = new_value
    return agent