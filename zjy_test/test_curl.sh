#curl -X POST "http://localhost:8101/generate" \
#-H "Content-Type: application/json" \
#-d '{"prompt": "<|im_start|>system
#You are a helpful assistant.
#<|im_end|>
#<|im_start|>user
#你好，Qwen2！请问今天的天气如何
#<|im_end|>
#<|im_start|>assistant", "max_tokens": 1000, "temperature": 0.1, "top_p": 0.8}'


curl -X POST "http://localhost:8101/v1/chat/completions" \
-H "Content-Type: application/json" \
-d @data.json