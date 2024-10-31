# README

Simulation and Evaluation Framework for AgentSense.

## Introduction

- structure

```bash
SENSE/
│
├── commands/ # sh command set for evaluation
├── outputs/ # evaluation outputs
├── configs/ # configs for simulation and evaluation
├── initialization.py # initialize a scenario setting according to config
├── simulation.py # simulation and evaluation for a single scenario
├── metric.py # compute metrics
├── run_eval.py # the main file for batch simulation and evaluation of multiple scenarios
└── utils/ 
    ├── data_utils.py
    ├── model_utils.py
    ├── label_role.py
    └── logger.py
```


## Preparation

```bash
conda create -n autogen python=3.10
conda activate autogen
pip install pyautogen
pip install autogen-agentchat~=0.2
pip install "fschat[model_worker,webui]"
pip install vllm
```

## Evaluation

### Example

#### API-based models  
- Interactions among single model-based agents  
example: commands/examples/gpt4o.sh  
*Remember to provide your API BASE and KEY.*

- Interactions among pair-wise model-based agents  
You need to first specify the driving models for sender agents and receiver agents to form a new json file for simulation and evaluation.  
    - Modify the sender model and receiver model, as well as their corresponding base_url and key in label_role.py.  
    - Run label_role.py to generate a new json file.  
    - Specify the input_math parameter as the path of the newly generated data, and specify the evaluation pattern parameter as' heter'.
example: commands/examples/heter_gpt4o_gpt35.sh 

#### Local models  

**Step1**：lauch the local servers  
You can lauch your local servers through fastchat, vllm or other engines.

- **vllm**: 
    - [doc](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
    - [using vllm in AutoGen](https://microsoft.github.io/autogen/docs/topics/non-openai-models/local-vllm)

- **fastchat**: 
    - [doc](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md)
    - [using fastchat in AutoGen](https://microsoft.github.io/autogen/blog/2023/07/14/Local-LLMs#clone-fastchat)


**Step2**: run the evaluations
Modify parameters and test corresponding models.  
example: sh commands/examples/llama3_8b.sh  

### Parameters

```bash
    # data
    parser.add_argument('--pattern', type=str, default='homo', help='interaction between homogeneous or heterogeneous agents, choose from [homo, heter]')
    parser.add_argument('--input_path', type=str, default="./data/final_data.jsonl", help='path of the file of scene setting')
    parser.add_argument('--config_dir', type=str, default="./configs/llama2_13b/", help='dir of generated cofig')
    parser.add_argument('--output_dir', type=str, default="./output/llama2_13b/", help='dir of output')
    # evaluation
    parser.add_argument('--prompt_template_path', type=str, default="./configs/prompt_template_hide.json", help='the path of prompt template for social agents and judge agents')
    parser.add_argument('--max_round', type=int, default=15, help='the max round of dialog')
    parser.add_argument('--speaker_selection_method', type=str, default='random', help='selection method of slecting the next speaker')
    parser.add_argument('--allow_repeat_speaker', type=bool, default=False, help='whether repeated speakers are allowed. must set False when only 2 agents are involved')
    # settings of social agents
    parser.add_argument('--model', type=str, default="Llama-2-13b-chat-hf", help='the driven models of agents')
    parser.add_argument('--base_url', type=str, default="http://0.0.0.0:8000/v1", help='url of API interface')
    parser.add_argument('--api_key', type=str, default="1234", help='key of API interface')
    parser.add_argument('--api_type', type=str, default="openai", help='type of API interface')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--max_tokens', type=int, default=128)
    # settings of judge agents
    parser.add_argument('--judge_config', type=str, default="./configs/judge_config.json", help='the paths of judge configs')    
    parser.add_argument('--option_mark', type=str, default='upper', help='option mark for multiple-choice questions in info reasoning evaluation')   
    parser.add_argument('--task_workers', type=int, default=4, help='num of parallel workers for simulation')
 
```

