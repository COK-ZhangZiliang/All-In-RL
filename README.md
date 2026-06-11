# 🚀 All-In-RL

## 📚 Learning & Implementation of Modern RL Algorithms

> 🧑‍💻 This is **ziliang**'s personal repository for learning and implementing modern reinforcement learning algorithms, with a focus on RL techniques applied to large language models (LLMs).

🎯 The goal of this repo is to provide minimal, self-contained, and readable implementations of representative RL / post-training algorithms, so that each method can be understood and reproduced from first principles.

## 🧩 Implemented Algorithms

| 🏷️ Algorithm                       | ✨ Features                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 📂 Code                                          |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| 🎓 **On-Policy Distillation (OPD)** | <ul><li>🐍 PyTorch + 🤗 HuggingFace Transformers implementation</li><li>🎲 Student generates rollouts **under its own policy** (on-policy)</li><li>🧊 Frozen teacher scores the same tokens</li><li>📐 Per-token **reverse KL** loss</li><li>🔤 Vocabulary compatibility check between teacher & student</li><li>📦 Supports HF datasets, local prompt files, or Python prompt lists</li><li>📊 Logs `reverse_kl`, `teacher_nll`, `student_nll`, `student_entropy`, `grad_norm`, `lr`</li></ul> | [📁 OPD/](./OPD) |

