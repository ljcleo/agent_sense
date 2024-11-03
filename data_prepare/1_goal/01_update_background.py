from pathlib import Path

from pydantic import BaseModel

from llm import LLM


class DialogLine(BaseModel):
    role: str
    content: str


class InputScenario(BaseModel):
    source: str | None
    episode: str
    scene_background: list[str]
    scenario_background: list[str]
    scenario: str
    dialog: list[DialogLine]


class Background(BaseModel):
    background: str


class OutputScenario(BaseModel):
    source: str | None
    episode: str
    background: str
    description: str
    dialog: list[DialogLine]


SYSTEM_PROMPT = """
You are an excellent writer good at analyzing story backgrounds.

You are given some information of a specific scenario in a story.
More specifically:

- The story is split into scenes, and you are given the background of
  each scene until the current one;
- The current scene is also split into scenarios, and you are given
  the background of each scenario until the current one;
- Finally, you are given the current scenario's description and dialog.

Write ONE paragraph to provide a DESCRIPTIVE background of the given scenario.
A good background should cover the information that sets up the scenario,
but does NOT reveal too many details from the scenario,
or include irrelevant details.
""".strip()

JSON_EXAMPLE = '{"background": "..."}'


def process(llm: LLM, input_line: str) -> str | None:
    scenario: InputScenario = InputScenario.model_validate_json(input_line)
    user_prompt: str = scenario.model_dump_json(exclude={"source", "episode"})

    return OutputScenario(
        source=scenario.source,
        episode=scenario.episode,
        background=llm(SYSTEM_PROMPT, JSON_EXAMPLE, user_prompt, Background).background,
        description=scenario.scenario,
        dialog=scenario.dialog,
    ).model_dump_json()


if __name__ == "__main__":
    data_dir = Path("data")
    i_dir: Path = data_dir / "01"
    o_dir: Path = data_dir / "02"
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
