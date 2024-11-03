from pathlib import Path
from typing import TypeVar

from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class LLMConfig(BaseModel):
    api_key: str
    base_url: str
    small_model: str
    large_model: str
    json_prompt: str
    fix_prompt: str

    @staticmethod
    def load(file: Path) -> "LLMConfig":
        return LLMConfig.model_validate_json(file.read_text(encoding="utf8"))


class LLM:
    def __init__(self, config_file: Path = Path("config.json")) -> None:
        self._config = LLMConfig.load(config_file)
        self._client = OpenAI(api_key=self._config.api_key, base_url=self._config.base_url)

    def __call__(
        self, system_prompt: str, json_example: str, user_prompt: str, output_type: type[T]
    ) -> T:
        while True:
            response: str = self._query_llm(
                f"{system_prompt}\n\n{self._config.json_prompt.format(example=json_example)}",
                user_prompt,
                self._config.large_model,
            )

            try:
                return output_type.model_validate_json(response)
            except Exception as e:
                print("parse failed:", e)
                print("try fixing ...")

                try:
                    return output_type.model_validate_json(
                        self._query_llm(
                            self._config.fix_prompt.format(example=json_example),
                            response,
                            self._config.small_model,
                        )
                    )
                except Exception as e:
                    print("fix failed:", e)

    def _query_llm(self, system_prompt: str, user_prompt: str, model: str) -> str:
        print("[SYSTEM PROMPT]", system_prompt, sep="\n", end="\n" + "-" * 40 + "\n")
        print("[USER PROMPT]", user_prompt, sep="\n", end="\n" + "-" * 40 + "\n")

        while True:
            try:
                response: ChatCompletion = self._client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=model,
                    response_format={"type": "json_object"},
                    seed=19260817,
                    temperature=0,
                )

                content: str | None = response.choices[0].message.content
                print("[RESPONSE CONTENT]", content, sep="\n", end="\n" + "-" * 40 + "\n")
                assert content is not None
                return content
            except Exception as e:
                print("get response failed:", e)
                continue
