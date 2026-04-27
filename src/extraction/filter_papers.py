#!/usr/bin/env python3
"""Filter the fetched papers to only real LLM/ML papers."""

# These are the papers from general_llm_papers_new.txt that are ACTUALLY about LLMs/ML,
# manually verified by reading the titles.

real_llm_papers = """2106.09063   — Specializing Multilingual Language Models: An Empirical Study (Chau et al.)
2201.07311   — Datasheet for the Pile (Biderman et al.)
2304.12244   — WizardLM: Empowering large pre-trained language models to follow complex instructions (Xu et al.)
2304.15010   — LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model (Gao et al.)
2305.13048   — RWKV: Reinventing RNNs for the Transformer Era (Peng et al.)
2309.10305   — Baichuan 2: Open Large-scale Language Models (Yang et al.)
2311.05741   — Efficiently Adapting Pretrained Language Models To New Languages (Csaki et al.)
2311.09227   — Open-Sourcing Highly Capable Foundation Models (Seger et al.)
2312.10793   — Demystifying Instruction Mixing for Fine-tuning Large Language Models (Wang et al.)
2312.15166   — SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling (Kim et al.)
2401.02385   — TinyLlama: An Open-Source Small Language Model (Zhang et al.)
2402.19427   — Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models (De et al.)
2403.02308   — Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures (Duan et al.)
2404.00934   — ChatGLM-RLHF: Practices of Aligning Large Language Models with Human Feedback (Hou et al.)
2404.03608   — Sailor: Open Language Models for South-East Asia (Dou et al.)
2404.05892   — Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence (Peng et al.)
2405.15032   — Aya 23: Open Weight Releases to Further Multilingual Progress (Aryabumi et al.)
2405.17428   — NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models (Lee et al.)
2405.19327   — MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series (Zhang et al.)
2406.11931   — DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence (DeepSeek-AI et al.)
2407.07726   — PaliGemma: A versatile 3B VLM for transfer (Beyer et al.)
2407.21783   — The Llama 3 Herd of Models (Grattafiori et al.)
2408.08152   — DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for RL and MCTS (Xin et al.)
2408.12570   — Jamba-1.5: Hybrid Transformer-Mamba Models at Scale (Team et al.)
2409.11272   — LOLA -- An Open-Source Massively Multilingual Large Language Model (Srivastava et al.)
2409.17146   — Molmo and PixMo: Open Weights and Open Data for State-of-the-Art VLMs (Deitke et al.)
2410.07073   — Pixtral 12B (Agrawal et al.)
2410.18982   — O1 Replication Journey: A Strategic Progress Report -- Part 1 (Qin et al.)
2412.01186   — SailCompass: Towards Reproducible and Robust Evaluation for Southeast Asian Languages (Guo et al.)
2412.04261   — Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier (Dang et al.)
2502.02737   — SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model (Allal et al.)
2601.01792   — HyperCLOVA X 8B Omni (Team)
2601.02346   — Falcon-H1R: Pushing the Reasoning Frontiers with a Hybrid Model (Team et al.)
2603.19220   — Nemotron-Cascade 2: Post-Training LLMs with Cascade RL (Yang et al.)
2506.13284   — AceReason-Nemotron 1.1: Advancing Math and Code Reasoning through SFT and RL (Liu et al.)
"""

# Papers already in the user's file (to avoid duplicates)
already_have = set([
    "2401.04088", "2402.00838", "2402.19173", "2403.04652", "2403.19887",
    "2405.04434", "2407.10671", "2407.10759", "2408.05147", "2408.12570",
    "2409.12191", "2412.01253", "2501.00656", "2512.13961", "2603.03975",
    "2603.11510", "2603.19220", "2604.03444",
    # From user's additional section
    "2307.09288", "2310.06825", "2403.17297", "2404.14219", "2407.21783",
    "2408.00118", "2408.03541", "2409.12817", "2410.02414", "2410.03185",
    "2410.05432", "2410.21199", "2411.02507", "2412.00285", "2412.02915",
    "2412.04156", "2501.07170", "2502.18886", "2504.12374", "2504.18345",
    "2505.09343", "2507.03821", "2508.14567",
    # From user's third section
    "2309.10305", "2311.00287", "2312.12960", "2401.06066", "2402.01601",
    "2404.09745", "2405.03437", "2405.08751", "2405.14219", "2406.09156",
    "2406.14143", "2407.16973", "2408.06234", "2408.19265", "2409.04321",
    "2409.07234", "2409.18297", "2410.01856", "2410.08834", "2411.08999",
    "2412.13961", "2501.00656",
    # From user's analysis section
    "2601.09631", "2602.15997", "2603.03459", "2603.07685", "2603.16197",
    "2603.21658", "2603.22816", "2604.08563", "2604.10135", "2604.16270",
    "2604.17761", "2604.21637", "2604.21751",
])

new_papers = []
for line in real_llm_papers.strip().split('\n'):
    arxiv_id = line.split('   —')[0].strip()
    if arxiv_id not in already_have:
        new_papers.append(line.strip())

print(f"New papers to add: {len(new_papers)}")
for p in new_papers:
    print(p)
