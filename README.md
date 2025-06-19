# Beyaznet LLM Inference

1. Download **gemma 3-4b** model from [this link](https://drive.google.com/drive/folders/1yGKLnu0gfpmcAao1tGpcrSg-8Ko35pEF) (download whole **google--gemma-3-4b-it** folder)

2. Install **requirements.txt**

3. Use following command to start vllm server on localhost. Wait until server is running (it may take 5-10 mins.)

```
vllm serve google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767 --dtype bfloat16
```

4. For model inference on given text files:

```
inference.py --model-path "google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767" \
 --system-prompt-path "system_prompt.txt" \
 --input-path "sample_input_paths.txt" \
 --output-path "sample_outputs"
```
