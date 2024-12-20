
#python serve_vllm_qwen.py
export CUDA_VISIBLE_DEVICES=7
python -m vllm.entrypoints.openai.api_server --tensor-parallel-size 1 \
--model "/mnt/tongyan.zjy/openlm/model/Qwen2.5-72B-Instruct-GPTQ-Int4" \
--served-model-name "qwen_72b_instruct"  \
--max-model-len 8000 \
--gpu-memory-utilization 0.7  \
--dtype=float16  \
 --enable-prefix-caching \
 --port 8101