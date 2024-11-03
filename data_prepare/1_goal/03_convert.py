from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from llm import LLM


class DialogLine(BaseModel):
    role: str
    content: str


class Character(BaseModel):
    name: str
    goals: list[str]


class InputScenario(BaseModel):
    source: str | None
    episode: str
    background: str
    description: str
    characters: list[Character]
    dialog: list[DialogLine]


class OutputScenario(BaseModel):
    background: str
    description: str
    characters: list[Character]


SYSTEM_PROMPT = """
You are an excellent psychologist good at designing social scenarios.

You are given a social scenario with background, description, and dialog.
You are also given the social goals of several major characters.

Set up a new social scenario involving only these **major characters**.
Each character's new social goals should appear **before** the scenario starts.

First, filter out contents from the background and description
that describes the detail of the scenario;
however, details of the beginning of the scenario can be kept.
Second, rewrite each character's social goals so that it:

- DOES NOT rely on other character's goals;
- DOES NOT include potential action the character will take;
- Uses infinitive verbs and third person pronouns.

Filter out social goals that cannot obey these criteria.
Modify the background/description to include more information if necessary.

Describe the background and description of the new scenario,
and list the new social goals of each major character.
""".strip()

JSON_EXAMPLE = (
    '{"background": "...", "description": "...", "characters": '
    '[{"name": "...", "goals": ["...", ...]}, ...]}'
)


def process(llm: LLM, input_line: str) -> str | None:
    scenario: InputScenario = InputScenario.model_validate_json(input_line)
    user_prompt: str = scenario.model_dump_json(exclude={"source", "episode"})
    buffer: list[str] = []

    for character, count in Counter(line.role for line in scenario.dialog).items():
        if count >= 2:
            buffer.append(character)

    if len(buffer) < 2:
        return None

    return llm(SYSTEM_PROMPT, JSON_EXAMPLE, user_prompt, OutputScenario).model_dump_json()


if __name__ == "__main__":
    data_dir = Path("data")
    i_dir: Path = data_dir / "03"
    o_dir: Path = data_dir / "04"
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
