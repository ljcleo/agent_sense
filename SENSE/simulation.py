"""
Base interface of simulation for a single scene
"""
from typing import List
import autogen
from autogen.agentchat.contrib.capabilities import transform_messages
from autogen.agentchat.contrib.capabilities.transforms import TextMessageContentName
from utils.model_utils import TextMessageTruncate
from pydantic import BaseModel
from initialization import load_scene, load_agent, load_judge_agent, load_groupchat, prepare_task_config, update_agent_llm_config
from metric import *
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class Simulation:
    def __init__(self, scene, agents, judge_agents, group_chat, chat_manager, output_dir):
        self.scene = scene
        self.agents = agents
        self.judge_agents = judge_agents
        self.group_chat = group_chat
        self.chat_manager = chat_manager
        self.output_dir = output_dir

        self.agent_dict = {}
        for agent in self.agents:
            self.agent_dict[agent.name] = agent
 
        for agent in self.judge_agents:
            # with 'judge_' prefix in its name
            self.agent_dict[agent.name] = agent
            
    @classmethod
    def from_task(cls, tasks_path: str, output_dir: str):
        task_config = prepare_task_config(tasks_path)
        # Build the scene
        scene = load_scene(task_config["scene"])
        # Build the agents
        agents = []
        for agent_config in task_config["agents"]:
            agent = load_agent(agent_config, task_config["scene"])
            agents.append(agent)

        judge_agents = []
        for agent_config in task_config["judge_agents"]:
            judge_agent = load_judge_agent(agent_config)
            judge_agents.append(judge_agent)
        
        # Build the group chat
        name_transform = TextMessageContentName(position="start", format_string="{name}: ")
        trunc_transform = TextMessageTruncate(trunc_symbol='\n')
        context_handling = transform_messages.TransformMessages(transforms=[name_transform, trunc_transform])
        for agent in agents:
            context_handling.add_to_agent(agent)
        for agent in judge_agents:
            context_handling.add_to_agent(agent)
        group_chat, chat_manager = load_groupchat(agents, task_config["groupchat"])

        return cls(scene, agents, judge_agents, group_chat, chat_manager, output_dir)

    def run(self,):
        # simulation
        start_agent = random.choice(self.agents)
        self.groupchat_result = start_agent.initiate_chat(
            self.chat_manager, message="Hi, there!"
        )
        # reset the temperature for evaluation
        for agent in self.agents:
            update_agent_llm_config(agent, 'temperature', self.judge_agents[0].llm_config['config_list'][0]['temperature'])
        # evaluation
        self.eval_goal()
        self.eval_info()
        # calculate metrics
        goal_metric = GoalMetric()
        info_metric = SingleChoiceMetric(self.scene['option_mark'])
        goal_res = goal_metric(self.goal_eval_res, self.judge_agents)
        if len(self.info_eval_res) and len(self.info_eval_res[list(self.info_eval_res.keys())[0]]):
            info_res = info_metric(self.info_eval_res, self.scene['info_question'])
        else:
            info_res = {}
        # save the results
        res = {
            "chat_history":self.groupchat_result.chat_history,
            "goal_answer":self.goal_eval_res,
            "goal_metrics":goal_res,
            "info_answer":self.info_eval_res,
            "info_metrics":info_res
        }
        with open(self.output_dir+'/'+str(self.scene['scene_id'])+'.json','w') as f:
            json.dump(res, f)
        return res

    
    def interview_agent(self, agent_name, question, chat_history):
        """interview an assigned agent """
        if isinstance(question, list) and len(question)==1:
            question = question[0]
        # acquire agent by the name
        agent = self.agent_dict[agent_name]
        # send msg to the assigned agent
        ans = agent.generate_reply(
            messages = chat_history+[{
                "content":question,
                "role":"user"
            }],
            sender = None,
        )
        return ans if type(ans)==str else ans['content']

    def eval_goal(self):
        """ evaluate goal completion """
        # chat history of social interaction
        chat_history = self.groupchat_result.chat_history
        goal_eval_res = {}
        for agent_name in self.scene['goal_question']:
            goal_eval_res[agent_name] = {}
            for i in range(len(self.scene['goal_question'][agent_name])):
                data = self.scene['goal_question'][agent_name][i]
                goal = data['goal']
                eval_question =data['eval_questions']
                for dim in eval_question: # self/other/judge
                    dim_res = []
                    for j in range(len(eval_question[dim])):
                        if dim in ['self','others']:
                            obj = eval_question[dim][j]['obj']
                            ques = eval_question[dim][j]['question']
                            ans = self.interview_agent(obj, ques, chat_history)
                            dim_res.append(ans)
                        elif dim =='judge':  
                            ques = eval_question[dim][j]['question']
                            with ThreadPoolExecutor(max_workers=len(self.judge_agents)) as executor:
                                results = {obj.name: executor.submit(self.interview_agent, obj.name, ques, chat_history) for obj in self.judge_agents}
                                dim_res.extend([results[obj.name].result() for obj in self.judge_agents])        
                        else:
                            raise NotImplementedError("unsupported eval dimension: {}".format(dim))    

                    if goal not in goal_eval_res[agent_name]:
                        goal_eval_res[agent_name][goal] = {}
                    goal_eval_res[agent_name][goal][dim] = dim_res
        self.goal_eval_res = goal_eval_res
        

    def eval_info(self):
        """ evaluate private info reasoning """
        # chat history of social interaction
        chat_history = self.groupchat_result.chat_history
        info_eval_res = {}
        for agent_name in self.scene['info_question']:
            info_eval_res[agent_name] = []
            for i in range(len(self.scene['info_question'][agent_name])):
                data = self.scene['info_question'][agent_name][i]
                ques = data['question_with_options']
                ans = self.interview_agent(agent_name, ques, chat_history)
                info_eval_res[agent_name].append(ans)
        self.info_eval_res = info_eval_res