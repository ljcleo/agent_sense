from pathlib import Path
from typing import Self
from llm import LLM
import json
from pydantic import BaseModel, Field, RootModel, model_validator
import re

def extract_json_block(text):
    pattern = r'```json\s*(.+?)\s*```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def read_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def write_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

OPT_GEN_PROMPT = '''You are a multiple-choice generator. Given a description of social scenario, a question and an answer, you need to generate 3 additional incorrect options. Incorrect options should be expressed in a similar way to the answer, but need to have completely different actual meanings so that they are sufficiently distinguishable from the answer.

###Description: {description}

###Question: {question}

###Answer: {answer}

Please return the results according to the following JSON structure:
```json
[{{"option1": "xxx", "option2": "xxx", "option3": "xxx"}}]
```
'''.strip()

if __name__ == '__main__':
    data_dir = Path('data')
    i_dir: Path = data_dir / '07'
    o_dir: Path = data_dir / '08'
    o_dir.mkdir(exist_ok=True)
    
    for input_file in i_dir.glob('*.jsonl'):
        data_05 = read_jsonl(input_file)
        output_data = []

        llm = LLM()

        for target_data in data_05:
            if target_data['private_info_label'] == 0:
                output_data.append(target_data)
                continue
            
            background = target_data['background']
            description = target_data['description']
            characters = target_data['characters']
            private_infos = target_data['private_infos']
            for idx in range(len(private_infos)):
                opts_output = llm(OPT_GEN_PROMPT.format(
                    description = description,
                    question = private_infos[idx]['question'],
                    answer = private_infos[idx]['answer']
                ))
                print(opts_output)
                print("-"*40)
                try:
                    opts = json.loads(opts_output)
                except:
                    opts = json.loads(extract_json_block(opts_output))
                private_infos[idx]['options'] = opts
            target_data['private_infos'] = private_infos
            output_data.append(target_data)

        output_filename = o_dir / input_file.name
        write_jsonl(output_data, output_filename)   