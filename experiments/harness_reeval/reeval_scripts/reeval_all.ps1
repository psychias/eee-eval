# Re-evaluation protocol - PowerShell
# Run evaluations under different harnesses to measure harness effect

# StarCoder2-15B on humaneval_plus: evalplus -> transformers
# Current score: 37.8 | Expected: 37.8
python eval_with_transformers.py --model bigcode/StarCoder2-15B --task humaneval+ --shots 0

# StarCoder2-15B on humaneval_plus: transformers -> evalplus
# Current score: 37.8 | Expected: 37.8
evalplus.evaluate --model bigcode/StarCoder2-15B --dataset humaneval --backend vllm

# starcoder2-15B-instruct-v0.1 on humaneval_plus: evalplus -> transformers
# Current score: 60.4 | Expected: 63.4
python eval_with_transformers.py --model bigcode/starcoder2-15B-instruct-v0.1 --task humaneval+ --shots 0

# starcoder2-15B-instruct-v0.1 on humaneval_plus: transformers -> evalplus
# Current score: 63.4 | Expected: 60.4
evalplus.evaluate --model bigcode/starcoder2-15B-instruct-v0.1 --dataset humaneval --backend vllm

# starcoder2-15B-instruct-v0.1 on mbpp_plus: evalplus -> transformers
# Current score: 65.1 | Expected: 61.2
python eval_with_transformers.py --model bigcode/starcoder2-15B-instruct-v0.1 --task mbpp+ --shots 3

# starcoder2-15B-instruct-v0.1 on mbpp_plus: transformers -> evalplus
# Current score: 61.2 | Expected: 65.1
evalplus.evaluate --model bigcode/starcoder2-15B-instruct-v0.1 --dataset mbpp --backend vllm

# StarCoder2-3B on humaneval_plus: evalplus -> transformers
# Current score: 27.4 | Expected: 27.4
python eval_with_transformers.py --model bigcode/StarCoder2-3B --task humaneval+ --shots 0

# StarCoder2-3B on humaneval_plus: transformers -> evalplus
# Current score: 27.4 | Expected: 27.4
evalplus.evaluate --model bigcode/StarCoder2-3B --dataset humaneval --backend vllm

# StarCoder2-7B on humaneval_plus: evalplus -> transformers
# Current score: 29.9 | Expected: 29.9
python eval_with_transformers.py --model bigcode/StarCoder2-7B --task humaneval+ --shots 0

# StarCoder2-7B on humaneval_plus: transformers -> evalplus
# Current score: 29.9 | Expected: 29.9
evalplus.evaluate --model bigcode/StarCoder2-7B --dataset humaneval --backend vllm

# Llama-3.1-Tulu-3-70B-SFT on ifeval: lm_eval -> vllm
# Current score: 80.51 | Expected: 80.51
python -m lm_eval --model vllm --model_args pretrained=allenai/Llama-3.1-Tulu-3-70B-SFT --tasks ifeval --num_fewshot 0 --batch_size auto

# Llama-3.1-Tulu-3-70B-SFT on ifeval: vllm -> lm_eval
# Current score: 80.51 | Expected: 80.51
lm_eval --model hf --model_args pretrained=allenai/Llama-3.1-Tulu-3-70B-SFT --tasks ifeval --num_fewshot 0 --batch_size auto --output_path results/allenai_Llama-3.1-Tulu-3-70B-SFT/ifeval

# Llama-3.1-Tulu-3-70B-SFT on bbh: lm_eval -> vllm
# Current score: 42.02 | Expected: 42.02
python -m lm_eval --model vllm --model_args pretrained=allenai/Llama-3.1-Tulu-3-70B-SFT --tasks bbh --num_fewshot 3 --batch_size auto

# Llama-3.1-Tulu-3-70B-SFT on bbh: vllm -> lm_eval
# Current score: 42.02 | Expected: 42.02
lm_eval --model hf --model_args pretrained=allenai/Llama-3.1-Tulu-3-70B-SFT --tasks bbh --num_fewshot 3 --batch_size auto --output_path results/allenai_Llama-3.1-Tulu-3-70B-SFT/bbh

# Llama-3.1-Tulu-3-70B-SFT on gpqa: lm_eval -> vllm
# Current score: 12.64 | Expected: 12.64
python -m lm_eval --model vllm --model_args pretrained=allenai/Llama-3.1-Tulu-3-70B-SFT --tasks gpqa_main_zeroshot --num_fewshot 0 --batch_size auto

# Llama-3.1-Tulu-3-70B-SFT on gpqa: vllm -> lm_eval
# Current score: 12.64 | Expected: 12.64
lm_eval --model hf --model_args pretrained=allenai/Llama-3.1-Tulu-3-70B-SFT --tasks gpqa_main_zeroshot --num_fewshot 0 --batch_size auto --output_path results/allenai_Llama-3.1-Tulu-3-70B-SFT/gpqa_main_zeroshot

# Llama-3.1-Tulu-3-70B-SFT on musr: lm_eval -> vllm
# Current score: 24.49 | Expected: 24.49
python -m lm_eval --model vllm --model_args pretrained=allenai/Llama-3.1-Tulu-3-70B-SFT --tasks musr --num_fewshot 0 --batch_size auto

# Llama-3.1-Tulu-3-70B-SFT on musr: vllm -> lm_eval
# Current score: 24.49 | Expected: 24.49
lm_eval --model hf --model_args pretrained=allenai/Llama-3.1-Tulu-3-70B-SFT --tasks musr --num_fewshot 0 --batch_size auto --output_path results/allenai_Llama-3.1-Tulu-3-70B-SFT/musr

# Llama-3.1-Tulu-3-70B-SFT on mmlu_pro: lm_eval -> vllm
# Current score: 40.27 | Expected: 40.27
python -m lm_eval --model vllm --model_args pretrained=allenai/Llama-3.1-Tulu-3-70B-SFT --tasks mmlu_pro --num_fewshot 5 --batch_size auto

# Llama-3.1-Tulu-3-70B-SFT on mmlu_pro: vllm -> lm_eval
# Current score: 40.27 | Expected: 40.27
lm_eval --model hf --model_args pretrained=allenai/Llama-3.1-Tulu-3-70B-SFT --tasks mmlu_pro --num_fewshot 5 --batch_size auto --output_path results/allenai_Llama-3.1-Tulu-3-70B-SFT/mmlu_pro

# Llama-3.1-Tulu-3-8B on ifeval: lm_eval -> vllm
# Current score: 82.67 | Expected: 82.55
python -m lm_eval --model vllm --model_args pretrained=allenai/Llama-3.1-Tulu-3-8B --tasks ifeval --num_fewshot 0 --batch_size auto

# Llama-3.1-Tulu-3-8B on ifeval: vllm -> lm_eval
# Current score: 82.55 | Expected: 82.67
lm_eval --model hf --model_args pretrained=allenai/Llama-3.1-Tulu-3-8B --tasks ifeval --num_fewshot 0 --batch_size auto --output_path results/allenai_Llama-3.1-Tulu-3-8B/ifeval

# Llama-3.1-Tulu-3-8B on bbh: lm_eval -> vllm
# Current score: 16.67 | Expected: 16.86
python -m lm_eval --model vllm --model_args pretrained=allenai/Llama-3.1-Tulu-3-8B --tasks bbh --num_fewshot 3 --batch_size auto

# Llama-3.1-Tulu-3-8B on bbh: vllm -> lm_eval
# Current score: 16.86 | Expected: 16.67
lm_eval --model hf --model_args pretrained=allenai/Llama-3.1-Tulu-3-8B --tasks bbh --num_fewshot 3 --batch_size auto --output_path results/allenai_Llama-3.1-Tulu-3-8B/bbh

# Llama-3.1-Tulu-3-8B on gpqa: lm_eval -> vllm
# Current score: 6.49 | Expected: 6.26
python -m lm_eval --model vllm --model_args pretrained=allenai/Llama-3.1-Tulu-3-8B --tasks gpqa_main_zeroshot --num_fewshot 0 --batch_size auto

# Llama-3.1-Tulu-3-8B on gpqa: vllm -> lm_eval
# Current score: 6.26 | Expected: 6.49
lm_eval --model hf --model_args pretrained=allenai/Llama-3.1-Tulu-3-8B --tasks gpqa_main_zeroshot --num_fewshot 0 --batch_size auto --output_path results/allenai_Llama-3.1-Tulu-3-8B/gpqa_main_zeroshot

# Llama-3.1-Tulu-3-8B on musr: lm_eval -> vllm
# Current score: 10.45 | Expected: 10.52
python -m lm_eval --model vllm --model_args pretrained=allenai/Llama-3.1-Tulu-3-8B --tasks musr --num_fewshot 0 --batch_size auto

# Llama-3.1-Tulu-3-8B on musr: vllm -> lm_eval
# Current score: 10.52 | Expected: 10.45
lm_eval --model hf --model_args pretrained=allenai/Llama-3.1-Tulu-3-8B --tasks musr --num_fewshot 0 --batch_size auto --output_path results/allenai_Llama-3.1-Tulu-3-8B/musr

# Llama-3.1-Tulu-3-8B on mmlu_pro: lm_eval -> vllm
# Current score: 20.3 | Expected: 20.23
python -m lm_eval --model vllm --model_args pretrained=allenai/Llama-3.1-Tulu-3-8B --tasks mmlu_pro --num_fewshot 5 --batch_size auto

# Llama-3.1-Tulu-3-8B on mmlu_pro: vllm -> lm_eval
# Current score: 20.23 | Expected: 20.3
lm_eval --model hf --model_args pretrained=allenai/Llama-3.1-Tulu-3-8B --tasks mmlu_pro --num_fewshot 5 --batch_size auto --output_path results/allenai_Llama-3.1-Tulu-3-8B/mmlu_pro

# Qwen2.5-72B-Instruct on gpqa: lm_eval -> unknown_papers_with_code
# Current score: 16.67 | Expected: 49.0
# No known command template for harness: unknown_papers_with_code

# Qwen2.5-72B-Instruct on gpqa: unknown_papers_with_code -> lm_eval
# Current score: 49.0 | Expected: 16.67
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-72B-Instruct --tasks gpqa_main_zeroshot --num_fewshot 0 --batch_size auto --output_path results/Qwen_Qwen2.5-72B-Instruct/gpqa_main_zeroshot
