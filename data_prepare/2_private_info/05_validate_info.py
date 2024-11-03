from pathlib import Path
from typing import Self
from llm import LLM
import json
from pydantic import BaseModel, Field, RootModel, model_validator

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

VALIDATE_INFO_PROMPT = '''You are an excellent psychologist who is good at analyzing the private information of each character in a social scenario. Private information refers to information that only the character knows and no one else knows. 

To determine whether there is private information, we need to check whether a specific character has information known to him/her, and whether the information exists in the background and description. Because the information in the background and description will be obtained by all characters, only when a specific character can obtain this information through its own goal and this information does not exist in the background and description, it indicates that the scene is a scene involving private information reasoning.

You only need to return *Yes* or *No* to confirm whether there is any private information. The following is the background information, description, main characters and corresponding social goals:

###Background: {background}

###Description: {description}

###Characters: {characters}'''.strip()
   
if __name__ == '__main__':
    data_dir = Path('data')
    # i_dir: Path = data_dir / '04'
    # o_dir: Path = data_dir / '05'
    i_dir: Path = data_dir / '05'
    o_dir: Path = data_dir / '06'
    o_dir.mkdir(exist_ok=True)
    
    for input_file in i_dir.glob('*.jsonl'):
        data_04 = read_jsonl(input_file)
        output_data = []

        llm = LLM()

        for target_data in data_04:
            background = target_data['background']
            description = target_data['description']
            characters = target_data['characters']

            validate_output = llm(VALIDATE_INFO_PROMPT.format(
                background=background,
                description=description,
                characters=characters
            ))
            target_data['private_info_label'] = 0 if 'no' in validate_output.lower() else 1
            output_data.append(target_data)

        output_filename = o_dir / input_file.name
        write_jsonl(output_data, output_filename)       