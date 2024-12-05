from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import qwen
import json
import os

from qwen_agent.llm import get_chat_model
system = """你是一个地理场景query结构化理解专家
你擅长对用户的请求进行结构化理解
给定一个问题，返回一个经过优化的结构化理解结果。
"""
prompt_user="""
例子:
输入：求姻缘
输出：寺庙 道观 教堂
输入：哪里买肉最便宜
输出：肉铺 市场 超市
"""
functions = [
    {
        "name": "query结构化理解",
        "description": "擅长在地理搜索场景下，把用户简单的query进行解析、改写与结构化理解",
        "parameters": {
          "type": "object",
          "properties": {
            "query改写": {
              "description": "把用户query改写成更加泛化明确的地理场景大类意图。省略所有筛选条件并输出用户可能想去的所有符合条件的通用地理场景。例如\"求姻缘\"改写成\"寺庙 道观 教堂\",\"哪里买肉最便宜\"改写成\"肉铺 市场 超市\"",
              "type": "string"
            },
            "城市过滤": {
              "description": "城市过滤器。Only use if explicitly specified.",
              "type": "string"
            },
            "最小距离过滤": {
              "description": "最小距离过滤器,单位是米。Only use if explicitly specified.",
              "type": "integer"
            },
            "最大距离过滤": {
              "description": "基于用户请求对于距离的需求度做最大距离过滤。最大距离分5档，100，1000，5000，10000，50000。景点类需求默认最大距离为50000，其他默认为10000。一般形容词越多可接受的距离需求越大",
              "type": "integer"
            },
            "最小评分过滤": {
              "description": "最小评分过滤器。Only use if explicitly specified.",
              "type": "float"
            },
            "最大评分过滤": {
	    "description": "最大评分过滤器。Only use if explicitly specified.",
	    "type": "float"
	    },
            "最小价格过滤": {
              "description": "最小价格过滤器.单位是元。Only use if explicitly specified.",
	      "type": "integer"
	    },
            "最大价格过滤": {
              "description": "最大价格过滤器.单位是元。Only use if explicitly specified.",
              "type": "integer"
            },
          },
          "required": [
          "query改写",
          "最大距离过滤"
          ],
        }
    }
    ]
def qr(query):
    messages = [{'role':'system','content':system},{'role': 'user', 'content': query}]
    llm = get_chat_model({
            'model': 'qwen_72b_instruct',
            'model_server': 'http://33.93.148.4:8000/v1/',
            'api_key': None,
            # 'model': 'Qwen-72B-Chat-Latest',
            # 'model_server': 'https://whale-wave.alibaba-inc.com/api/v2/services/aigc/text-generation/chat/completions',
            # 'api_key': 'BZHXK8UQ0W',
            'generate_cfg': {
                    'top_k':1,
                    'temperature':0.1,
                    'top_p':0.8,
                     'seed':1024
                }
        })
    responses = llm.chat(messages=messages,functions=functions,stream=False)
    return responses[-1]
print(qr("低于100元卖肉"))