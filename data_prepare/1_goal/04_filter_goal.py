from pathlib import Path

from pydantic import BaseModel

from llm import LLM


class Character(BaseModel):
    name: str
    goals: list[str]


class Scenario(BaseModel):
    background: str
    description: str
    characters: list[Character]


class Input(BaseModel):
    background: str
    description: str
    characters: str
    current_character: str
    goals: list[str]
    current_goal: str


class Output(BaseModel):
    examination: str
    update: str


SYSTEM_PROMPT = """
You are an excellent psychologist good at analyzing social goals.

You are given the social goals of a character in a designed social scenario.
You are provided the background, description and character lists.

Now, for the specified goal, check if it needs to be rewritten or removed
due to any of these reasons:

1. The goal directly involves characters not participating in the scenario,
   e.g. 'deal with the client' (if 'client' is not in the list of characters);
2. The goal requires information not provided in the background or description,
   e.g. 'describe the plan' (if the plan already exists but not provided);
3. The goal is a physical action, e.g. 'fix the television';
4. The goal is too abstract to evaluate, e.g. 'navigate professional challenges';
5. The goal is too subjective to evaluate, e.g. 'maintain dignity';
6. The goal is meaningless to evaluate, e.g. 'join the conversation'.

Write a detailed paragraph to examine the social goal.
Compare it with each of the criteria above.
If the goal matches one or more criteria above,
check if you can rewrite the goal to avoid them.
You should still remove the goal if this is not possible.

Based on your examination, write an updated version of the goal:
- If the goal is valid, return the original goal.
- If the goal can be rewritten, return the rewritten goal.
- If the goal needs to be removed, return an empty string.

Finally, any returned goal (if any) should be formatted into 'To xxx.',
e.g. 'To share his/her discovery.' (including the final period).
""".strip()

JSON_EXAMPLE = '{"examination": "...", "update": "..."}'


def process(llm: LLM, input_line: str) -> str | None:
    scenario: Scenario = Scenario.model_validate_json(input_line)
    character_list: str = "/".join(character.name for character in scenario.characters)

    for character in scenario.characters:
        for _ in range(2):
            buffer: list[str] = []

            for goal in character.goals:
                user_prompt: str = Input(
                    background=scenario.background,
                    description=scenario.description,
                    characters=character_list,
                    current_character=character.name,
                    goals=character.goals,
                    current_goal=goal,
                ).model_dump_json()

                output = llm(SYSTEM_PROMPT, JSON_EXAMPLE, user_prompt, Output)
                if output.update != "":
                    buffer.append(output.update)

            if len(buffer) == 0:
                return None

            character.goals = buffer

    return scenario.model_dump_json()


if __name__ == "__main__":
    data_dir = Path("data")
    i_dir: Path = data_dir / "04_merge"
    o_dir: Path = data_dir / "05"
    o_dir.mkdir(exist_ok=True)
    llm = LLM()

    for file in i_dir.iterdir():
        buffer: list[str] = []

        with file.open(encoding="utf8") as f:
            for line in f:
                result: str | None = process(llm, line)
                if result is not None:
                    buffer.append(result)

        with (o_dir / file.name).open("w", encoding="utf8") as f:
            print(*buffer, sep="\n", file=f)
