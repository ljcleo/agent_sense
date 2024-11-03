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

GEN_INFO_PROMPT = '''
You are good at writing questions for specific roles based on a social scenario. Below you will be provided with background information, a description of the current scene, and the goals of each of the main characters.

###Background: {background}

###Description: {description}

###Characters: {characters}

Please try to give some questions that the target character (in the following JSON format content, 'role' is used to refer to) can answer, but other characters will have difficulty answering before the interaction. These questions should strictly contain information that the target character knows, but is beyond the knowledge of other characters, so other characters cannot answer them at first. Specifically, the information required for these questions cannot appear in the background and description, because other characters will obtain this part as information. Questions cannot be expressed in the second person because the questions will eventually be used to ask other characters. For example, when the target character of a question is Rose, "Rose, why did you ..." is not a good question, but should be written as "Why did Rose ..."

Please provide a statement (in the following JSON format content, 'explanation' is used to refer to) that explains why the target character can answer the question, but other characters cannot. The statement should be objective factual information presented in the script, and should not mention the question, so it cannot appear in a sentence structure like "This question is ...".

Please provide the correct answer to the question, and the answer can be found in the information given.

Please use casual language as much as possible, and try to ask questions in the third person, such as "What is Jason's true identity?". Please answer in English. Please return the results according to the following JSON structure:
```json
[{{"role": str, "question": str, "explanation": str, "answer": str}}, {{"role": str, "question": str, "explanation": str, "answer": str}}]
```
'''.strip()

if __name__ == '__main__':
    data_dir = Path('data')
    i_dir: Path = data_dir / '06'
    o_dir: Path = data_dir / '07'
    o_dir.mkdir(exist_ok=True)
    
    for input_file in i_dir.glob('*.jsonl'):
        data_05 = read_jsonl(input_file)
        output_data = []

        llm = LLM()

        for target_data in data_05:
            if target_data['private_info_label'] == 0:
                target_data['private_infos'] = []
                output_data.append(target_data)
                continue

            background = target_data['background']
            description = target_data['description']
            characters = target_data['characters']

            info_output = llm(GEN_INFO_PROMPT.format(
                background=background,
                description=description,
                characters=characters
            ))
            print(info_output)
            try:
                private_infos = json.loads(info_output)
            except:
                private_infos = json.loads(extract_json_block(info_output))
            target_data['private_infos'] = private_infos
            output_data.append(target_data)

        output_filename = o_dir / input_file.name
        write_jsonl(output_data, output_filename)   