from pathlib import Path
from openai import OpenAI
from openai.types.chat import ChatCompletion


class LLMConfig:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model


    @staticmethod
    def load(file: Path) -> "LLMConfig":
        config_data = eval(file.read_text(encoding="utf8"))
        return LLMConfig(**config_data)


class LLM:
    def __init__(self, config_file: Path = Path("config.json")) -> None:
        self._config = LLMConfig.load(config_file)
        self._client = OpenAI(api_key=self._config.api_key, base_url=self._config.base_url)

    def __call__(self, prompt: str) -> str:
        return self._query_llm(prompt, self._config.model)

    def _query_llm(self, user_prompt: str, model: str) -> str:
        while True:
            try:
                response: ChatCompletion = self._client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": user_prompt},
                    ],
                    model=model,
                    response_format={"type": "text"},
                    seed=19260817,
                    temperature=0,
                )

                content: str | None = response.choices[0].message.content
                assert content is not None
                return content
            except Exception as e:
                print("get response failed:", e)
                continue
