from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from llm import LLM


class DialogLine(BaseModel):
    role: str
    content: str


class InputScenario(BaseModel):
    source: str | None
    episode: str
    background: str
    description: str
    dialog: list[DialogLine]


class Character(BaseModel):
    name: str
    goals: list[str]


class OutputScenario(BaseModel):
    source: str | None
    episode: str
    background: str
    description: str
    characters: list[Character]
    dialog: list[DialogLine]


SYSTEM_PROMPT = """
You are an excellent psychologist good at understanding social goals and needs.

You are given a social scenario with its background, description, and dialog.
For the specific character of **{character}**, identify their social goals.
Social goals typically fall into one of these categories:

- Exchange information with others;
- Build relationship with others;
- Maintain relationship or provide emotional support;
- Identify themselves with a group;
- Co-operate with others;
- Compete with others;
- Resolve conflicts.

Social goals should be objective, specific and clear;
whether the character has achieved them should be observable.

The character can have one single goal or multiple independent goals
in the scenario; find and list all of them.
For each goal, write a sentence to describe the goal.
Use infinitive verbs and third person pronouns.
""".strip()

JSON_EXAMPLE = '{"name": "...", "goals": ["...", ...]}'


def process(llm: LLM, input_line: str) -> str | None:
    scenario: InputScenario = InputScenario.model_validate_json(input_line)
    user_prompt: str = scenario.model_dump_json(exclude={"source", "episode"})
    buffer: list[str] = []

    for character, count in Counter(line.role for line in scenario.dialog).items():
        if count >= 2:
            buffer.append(character)

    if len(buffer) < 2:
        return None

    return OutputScenario(
        source=scenario.source,
        episode=scenario.episode,
        background=scenario.background,
        description=scenario.description,
        characters=[
            Character(
                name=character,
                goals=llm(
                    SYSTEM_PROMPT.format(character=character), JSON_EXAMPLE, user_prompt, Character
                ).goals,
            )
            for character in buffer
        ],
        dialog=scenario.dialog,
    ).model_dump_json()


if __name__ == "__main__":
    data_dir = Path("data")
    i_dir: Path = data_dir / "02"
    o_dir: Path = data_dir / "03"
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
