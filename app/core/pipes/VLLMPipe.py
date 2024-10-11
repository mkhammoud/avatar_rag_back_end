from typing import Callable

from openai import OpenAI

from app.core.Pipeline import Pipe
import subprocess


class VLLMPipe(Pipe):
    def __init__(self, model):
        self.model = model
        self.client = None
        self.process = None
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )
        if not self.is_hosting():
            raise Exception('VLLM is not working. Please try again after starting the VLLM server.')

    def is_hosting(self) -> bool:
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello!"}]
            )
            return True
        except Exception as ex:
            return False

    def exec(self, arg) -> any:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": arg}]
        )
        return completion.choices[0].message.content

    def end(self):
        pass
