"""
Calculate metrics for evaluation
"""

import re
import numpy as np
import statistics

class GoalMetric(object):
    def __init__(self):
        super().__init__()

    def judge(self, answer):
        # judge a single record
        if 'Yes' in answer:
            return 1
        else:
            return 0

    def __call__(self, goal_eval_res, judge_agents):
        res = {}
        for agent_name in goal_eval_res:
            res[agent_name] = {}
            goals = list(goal_eval_res[agent_name].keys())
            for goal in goals:
                res[agent_name][goal] = {}
                for dim in goal_eval_res[agent_name][goal]: # self/others/judge
                    dim_res = []
                    for ans in goal_eval_res[agent_name][goal][dim]: 
                        score = self.judge(ans)
                        dim_res.append(score)
                    if dim in ['self','others']:
                        dim_res = np.mean(dim_res)
                        res[agent_name][goal][dim] = dim_res # avg of self/other
                    else:
                        judge_avg = np.mean(dim_res)
                        judge_mode = statistics.mode(dim_res)
                        dim_res.append(judge_avg)
                        dim_res.append(judge_mode)
                        res[agent_name][goal][dim] = dim_res # record by judge model
            # avg by goals
            if 'self' in res[agent_name][goals[0]]:
                res[agent_name]['self'] = np.mean([res[agent_name][goal]['self'] for goal in goals])
            if 'others' in res[agent_name][goals[0]]:
                res[agent_name]['others'] = np.mean([res[agent_name][goal]['others'] for goal in goals])
            for i in range(len(judge_agents)):
                res[agent_name][judge_agents[i].name] = np.mean([res[agent_name][goal]['judge'][i] for goal in goals])
            res[agent_name]['judge_avg'] = np.mean([res[agent_name][goal]['judge'][-2] for goal in goals])    
            res[agent_name]['judge_majority'] = np.mean([res[agent_name][goal]['judge'][-1] for goal in goals])
        # avg by agents
        agent_names = list(goal_eval_res.keys())
        if 'self' in res[agent_name]:
            res['self'] = np.mean([res[agent_name]['self'] for agent_name in agent_names])
        if 'others' in res[agent_name]:
            res['others'] = np.mean([res[agent_name]['others'] for agent_name in agent_names])
        for i in range(len(judge_agents)):
            res[judge_agents[i].name] = np.mean([res[agent_name][judge_agents[i].name] for agent_name in agent_names])
        if 'judge_avg' in res[agent_name]:
            res['judge_avg'] = np.mean([res[agent_name]['judge_avg'] for agent_name in agent_names])
        if 'judge_majority' in res[agent_name]:
            res['judge_majority'] = np.mean([res[agent_name]['judge_majority'] for agent_name in agent_names])
        return res


class SingleChoiceMetric(object):
    def __init__(self, alphabet='all'):
        if alphabet == 'all':
            re_format = '\([1-9A-Za-z]\)'
        elif alphabet == 'upper':
            re_format = '\([A-Z]\)'
        elif alphabet == 'lower':
            re_format = '\([a-z]\)'
        elif alphabet == 'number':
            re_format = '\([1-9]\)'
        self.re_format = re_format
        ab = ['ABCDEFGHIJKLMNOPQRSTUVWXYZ',
              'abcdefghijklmnopqrstuvwxyz',
              '123456789']
        self.ab_map = {}
        for ab_item in ab:
            self.ab_map.update({k:i for i,k in enumerate(ab_item)})

    def judge(self, prediction, answer, options=None):
        # judge a single record
        # match the option item
        patterns = re.findall(self.re_format, prediction)
        if len(patterns) == 0:
            if len(prediction) == 1 or (len(prediction)==2 and prediction[1]=='.'):
                prediction = prediction.replace('.','')
                # only one single character in the output
                flag = False
                if prediction in self.ab_map:
                    flag = True
                    pred_index = self.ab_map[prediction]
                    # tmp_pred = self.ab_map[]
                if flag:
                    return int(pred_index==int(answer)), pred_index
                else:
                    return 0, None
            if options is not None and len(prediction):
                # match the content of options
                clean_pattern = '[A-Z]\)|[A-Z]\.'
                if re.match(clean_pattern+'.+',prediction): #startsiwth
                    prediction = re.sub(clean_pattern, '',prediction).strip()
                flag = False
                pred_index = None
                for i,opt in enumerate(options):
                    other_opt = list(options)
                    other_opt.remove(opt)
                    if opt == prediction or (opt in prediction and sum([int(neg_opt in prediction) for neg_opt in other_opt])==0) or (prediction in opt and sum([int(prediction in neg_opt) for neg_opt in other_opt])==0):
                        # no answer mark but exact match; contain the answer and exclude other options
                        flag = True
                        pred_index = i
                        break
                if flag:
                    return int(pred_index==int(answer)), pred_index
                else:
                    return 0, None
            else:
                # format error
                return 0, None
        else:
            pred = self.ab_map[patterns[0][1]] # the first (*)
            if pred == int(answer):
                return 1, pred
            else:
                return 0, pred
    
    def __call__(self, info_eval_res, info_data):
        res = {}
        for agent_name in info_eval_res:
            agent_res = []
            for i in range(len(info_eval_res[agent_name])):
                pred = info_eval_res[agent_name][i]
                ans = info_data[agent_name][i]['answer_label']
                options = info_data[agent_name][i]['options']
                m, pred = self.judge(pred, ans, options)
                agent_res.append(m)
            res[agent_name] = np.mean(agent_res)
        # avg over agents
        res['avg'] = np.mean(list(res.values()))
        return res
