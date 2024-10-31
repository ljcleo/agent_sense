"""
label the roles of sender and receiver
"""

import json
from openai import OpenAI
import time
from tqdm import tqdm
import random
import copy

    
def prepare_heter_data(data, sender_model, sender_base_url, sender_api_key, sender_api_type,
                       receiver_model, receiver_base_url, receiver_api_key, receiver_api_type):
    data = [d for d in data if set(d['roles'].values())==set(['sender','receiver'])]
    print('# of scenes: ',len(data))
    for i in range(len(data)):
        role_dict = data[i]['roles']
        for j in range(len(data[i]['characters'])):
            name = data[i]['characters'][j]['name']
            if role_dict[name]=='sender':
                data[i]['characters'][j]['model'] = sender_model
                data[i]['characters'][j]['base_url'] = sender_base_url
                data[i]['characters'][j]['api_key'] = sender_api_key
                data[i]['characters'][j]['api_type'] = sender_api_type
            else:
                data[i]['characters'][j]['model'] = receiver_model
                data[i]['characters'][j]['base_url'] = receiver_base_url
                data[i]['characters'][j]['api_key'] = receiver_api_key
                data[i]['characters'][j]['api_type'] = receiver_api_type     
    return data           
    

if __name__=="__main__":
    data = open("../data/data_with_role.jsonl","r").readlines()
    data = [json.loads(d) for d in data]
    sender_model= "gpt-4o"
    sender_base_url= "xxxx"
    sender_api_key= "xxxx"
    sender_api_type="openai"
    receiver_model= "gpt-3.5-turbo"
    receiver_base_url= "xxxx"
    receiver_api_key=  "xxxx"
    receiver_api_type="openai"
    data = prepare_heter_data(data, sender_model, sender_base_url, sender_api_key, sender_api_type,
                       receiver_model, receiver_base_url, receiver_api_key, receiver_api_type)
    with open('../data/heter/{}_{}.jsonl'.format(sender_model,receiver_model),'w') as f:
        for line in data:
            f.write(json.dumps(line)+'\n')