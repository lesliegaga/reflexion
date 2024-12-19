from typing import Union, Literal
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.schema import (
    HumanMessage
)
import os
os.environ['OPENAI_API_KEY']='sk'

class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        # Determine model type from the kwargs
        model_name = kwargs.get('model_name', 'gpt-3.5-turbo')
        kwargs['openai_api_base'] = 'http://localhost:8101/v1'
        if model_name.split('-')[0] == 'text':
            self.model = OpenAI(*args, **kwargs)
            self.model_type = 'completion'
        else:
            self.model = ChatOpenAI(*args, **kwargs)
            self.model_type = 'chat'

    def __call__(self, prompt: str):
        if self.model_type == 'completion':
            return self.model(prompt)
        else:
            return self.model(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            ).content

if __name__ == '__main__':
    llm = AnyOpenAILLM(
    temperature=0,
    max_tokens=100,
    model_name="qwen_72b_instruct",
    model_kwargs={"stop": "\n"},
    openai_api_key=os.environ['OPENAI_API_KEY'])
    response= llm("你好，Qwen2！请问今天的天气如何")
    print("生成的响应：")
    print(response)