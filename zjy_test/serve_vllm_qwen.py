import asyncio
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# 定义请求体
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 10000
    temperature: float = 0.1
    top_k: float = 1.0
    top_p: float = 0.8


# 初始化 FastAPI 应用
app = FastAPI(title="Qwen2 vLLM Service")

# 初始化 LLM 模型（全局变量，避免每次请求都加载模型）
# 替换 'path/to/qwen2/model' 为实际模型路径或 Hugging Face 模型 ID

model = LLM(model="/mnt/tongyan.zjy/openlm/model/Qwen2.5-72B-Instruct-GPTQ-Int4",
            tensor_parallel_size = 2, dtype = 'float16', disable_log_stats = False, max_model_len = 512 * 50,
            gpu_memory_utilization = 0.9
            )


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    try:
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            # 其他参数可根据需求调整
        )

        # 使用 vLLM 生成文本
        outputs = await model.generate(request.prompt, sampling_params)
        generated_text = outputs.generations[0].text  # 获取第一个生成结果

        return {"generated_text": generated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 启动 API 服务，监听 0.0.0.0:8000
    uvicorn.run(app, host="0.0.0.0", port=8101)