"""Supported models. Newest first within each company. PRs welcome."""

MODELS = {
    "Qwen": [
        # 3.6 (Apr 2026)
        "Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B-FP8", "Qwen/Qwen3.6-27B",
        # 3.5 (Feb 2026)
        "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-122B-A10B", "Qwen/Qwen3.5-397B-A17B",
        # 3
        "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B",
        "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-8B-Thinking",
        # 2.5
        "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-72B-Instruct",
        # coder + reasoning
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/QwQ-32B",
    ],
    "Meta": [
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct",
    ],
    "Mistral": [
        # 2026
        "mistralai/Mistral-Small-4-119B-2603",
        "mistralai/Mistral-Small-4-119B-2603-NVFP4",
        "mistralai/Mistral-Large-3-675B-Instruct-2512",
        "mistralai/Mistral-Large-3-675B-Base-2512",
        # 2025 still current
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "mistralai/Mistral-Large-Instruct-2411",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Codestral-22B-v0.1",
        "mistralai/Devstral-Small-2505",
        "mistralai/Magistral-Small-2506",
    ],
    "Google": [
        # Gemma 4 (Apr 2026) — multimodal, 256K ctx, reasoning mode
        "google/gemma-4-31B-it", "google/gemma-4-31B",
        "google/gemma-4-26B-A4B-it", "google/gemma-4-26B-A4B",
        "google/gemma-4-E4B-it", "google/gemma-4-E4B",
        "google/gemma-4-E2B-it", "google/gemma-4-E2B",
        # Gemma 3
        "google/gemma-3-1b-it", "google/gemma-3-4b-it",
        "google/gemma-3-12b-it", "google/gemma-3-27b-it",
        "google/gemma-3n-E2B-it", "google/gemma-3n-E4B-it",
        "google/gemma-2-9b-it", "google/gemma-2-27b-it",
    ],
    "Microsoft": [
        "microsoft/Phi-4-mini-instruct", "microsoft/Phi-4-mini-reasoning",
        "microsoft/phi-4", "microsoft/Phi-4-reasoning",
        "microsoft/Phi-3.5-mini-instruct", "microsoft/Phi-3.5-MoE-instruct",
    ],
    "DeepSeek": [
        # V4 preview (Apr 2026)
        "deepseek-ai/DeepSeek-V4-Pro",
        "deepseek-ai/DeepSeek-V4-Pro-Base",
        "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-ai/DeepSeek-V4-Flash-Base",
        # V3.2
        "deepseek-ai/DeepSeek-V3.2",
        "deepseek-ai/DeepSeek-V3.2-Speciale",
        "deepseek-ai/DeepSeek-V3.2-Exp",
        "nvidia/DeepSeek-V3.2-NVFP4",
        # R1/V3
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3-0324",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    ],
    "GLM": [
        # 5.1 (2026)
        "zai-org/GLM-5.1", "zai-org/GLM-5.1-FP8",
        "zai-org/GLM-5", "zai-org/GLM-4.7-Flash", "zai-org/GLM-4.7",
        "zai-org/GLM-4.6", "zai-org/GLM-4.5", "zai-org/GLM-4.5-Air",
    ],
    "Moonshot": [
        # K2.6 (Apr 2026) — 32B active / 1T total MoE, 300-sub-agent orchestration
        "moonshotai/Kimi-K2.6",
        "moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2-Instruct",
        "moonshotai/Kimi-Dev-72B", "moonshotai/Moonlight-16B-A3B-Instruct",
    ],
    "MiniMax": [
        # M2.7 (Apr 2026) — self-evolution training loop
        "MiniMaxAI/MiniMax-M2.7",
        "MiniMaxAI/MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.1",
        "MiniMaxAI/MiniMax-M2", "MiniMaxAI/MiniMax-Text-01-hf",
    ],
    "Nvidia": [
        # 2026 Nemotron 3 refresh
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        "nvidia/OpenReasoning-Nemotron-32B",
    ],
    "IBM Granite": [
        # 2026 releases
        "ibm-granite/granite-4.1-8b",
        "ibm-granite/granite-4.0-3b-vision",
        "ibm-granite/granite-4.0-1b-speech",
    ],
    "Tencent Hunyuan": [
        "tencent/Hy3-preview",          # 21B active / 295B total MoE, Apr 2026
        "tencent/HY-OmniWeaving",       # multimodal, Apr 2026
    ],
    "StepFun": [
        "stepfun-ai/Step-3.5-Flash",    # 11B active MoE, 256K ctx
        "stepfun-ai/Step-3.5-Flash-Base",
    ],
    "Baidu ERNIE": [
        "baidu/ERNIE-4.5-VL-28B-A3B-Thinking",
    ],
    "Cohere": [
        "CohereLabs/c4ai-command-a-03-2025",
        "CohereLabs/c4ai-command-r7b-12-2024",
        "CohereLabs/aya-expanse-8b", "CohereLabs/aya-expanse-32b",
    ],
    "Yi": ["01-ai/Yi-1.5-9B-Chat", "01-ai/Yi-1.5-34B-Chat", "01-ai/Yi-Coder-9B-Chat"],
    "InternLM": ["internlm/internlm3-8b-instruct", "internlm/internlm2_5-7b-chat"],
    "Falcon": ["tiiuae/Falcon3-7B-Instruct", "tiiuae/Falcon3-10B-Instruct"],
    "AllenAI": ["allenai/Olmo-3-1125-32B", "allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3.1-32B-Instruct"],
}
