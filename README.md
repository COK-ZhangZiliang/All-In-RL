# 🚀 All-In-RL

## 📚 Learning & Implementation of Modern RL Algorithms

> 🧑‍💻 This is **ziliang**'s personal repository for learning and implementing modern reinforcement learning algorithms, with a focus on RL techniques applied to large language models (LLMs).

🎯 The goal of this repo is to provide minimal, self-contained, and readable implementations of representative RL / post-training algorithms, so that each method can be understood and reproduced from first principles.

## 🤖 AI-Native

This repository is **AI-native**: it is meant to be developed with AI coding agents. The conventions, architecture, and contribution workflow live in [AGENTS.md](./AGENTS.md), which an AI agent reads and follows to add new algorithms or modify existing ones while keeping the codebase minimal and consistent.

## 🧩 Implemented Algorithms

| 🏷️ Algorithm                       | ✨ Features                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 📂 Code                                          |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| 🎓 **On-Policy Distillation (OPD)** | <ul><li>🐍 PyTorch + 🤗 HuggingFace Transformers implementation</li><li>🎲 Student generates rollouts **under its own policy** (on-policy)</li><li>🧊 Frozen teacher scores the same tokens</li><li>📐 Per-token **reverse KL** loss</li><li>🔤 Vocabulary compatibility check between teacher & student</li><li>📦 Supports HF datasets, local prompt files, or Python prompt lists</li><li>📊 Logs `reverse_kl`, `teacher_nll`, `student_nll`, `student_entropy`, `grad_norm`, `lr`</li></ul> | [📁 OPD/](./OPD) |
| 🌐 **Multi-Teacher On-Policy Distillation (MOPD)** | <ul><li>🧑‍🏫 Multi-teacher extension of OPD (MiMo-V2-Flash recipe)</li><li>🗂️ Per-prompt `domain` field routes each rollout to its **domain expert teacher**</li><li>📐 Per-token **reverse KL** towards the routed teacher</li><li>⚖️ **Truncated importance weights** correct the sampling/training temperature mismatch</li><li>🔤 Vocab check across the whole teacher pool</li><li>📊 Logs `reverse_kl`, `teacher_nll`, `student_nll`, `student_entropy`, `is_weight_mean`, `grad_norm`, `lr`</li></ul> | [📁 MOPD/](./MOPD) |
| 🎯 **Proximal Policy Optimization (PPO)** | <ul><li>🐍 PyTorch + 🤗 HuggingFace Transformers implementation</li><li>🧩 Classic four-model RLHF: **actor / critic / reference / reward**</li><li>🎲 On-policy rollouts + per-token **KL-to-reference** reward shaping</li><li>📉 **GAE** advantages with a trainable **critic** (separate backbone *or* shared value head)</li><li>✂️ **PPO-clipped** policy surrogate + **clipped value** loss, multi-epoch minibatch reuse</li><li>🏆 Switchable reward: frozen **reward model** *or* **verifiable** rule-based reward (RLVR)</li><li>📊 Logs `policy_loss`, `value_loss`, `entropy`, `clip_frac`, `approx_kl`, `score`, `kl`, `grad_norm`, `lr`</li></ul> | [📁 PPO/](./PPO) |

