import asyncio
from vllm import LLM, CompletionConfig
from fastapi import FastAPI, Request
import uvicorn

# 初始化 FastAPI 应用
app = FastAPI()

# 初始化 vLLM 模型
llm = LLM(model_path="/mnt/tongyan.zjy/openlm/model/Qwen2.5-72B-Instruct-GPTQ-Int4")  # 替换为你的模型路径

# 配置生成参数
completion_config = CompletionConfig(
    temperature=0.1,
    top_k=1.0,
    max_tokens=10000,
    top_p=0.8,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)


@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    # 生成文本
    completions = await llm.generate(prompt, completion_config)
    response = completions["choices"][0]["text"]

    return {"generated_text": response}


if __name__ == "__main__":
    # 启动 Uvicorn 服务器
    uvicorn.run(app, host="0.0.0.0", port=8101)