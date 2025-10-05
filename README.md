# 🧠 Project Overview

This project demonstrates how to **train a Reward Model** — a key component in **Reinforcement Learning from Human Feedback (RLHF)** — using **Hugging Face’s `trl`** library.  
It teaches a model to score which response is more human-preferred, based on the **Anthropic HH-RLHF dataset**.

> 🧩 This is the same kind of model used to align LLMs like ChatGPT during training.

---

## ⚙️ Key Components

- **Base Model:** `Qwen/Qwen3-0.6B`  
- **Dataset:** `Dahoas/full-hh-rlhf`  
- **Frameworks:** `transformers`, `trl`, `datasets`, `torch`  
- **Trainer:** `RewardTrainer` from Hugging Face TRL  

---

## 🧩 Pipeline Steps

1. **Load dataset** — Preference pairs of *(prompt, chosen, rejected)*.  
2. **Preprocess & tokenize** — Combine *prompt + responses* and set `max_length`.  
3. **Initialize Reward Model** — `AutoModelForSequenceClassification` with `num_labels=1`.  
4. **Train using RewardTrainer** — Optimizes the model to prefer “chosen” responses.  
5. **Evaluate & Inference** — Compare reward scores on custom prompts.  

---

## 📊 Results Example

**Prompt:** Why is the sky blue?  
**Good:** Because it has refraction.  
**Bad:** Because it is not green.  

Reward(good) = 0.92
Reward(bad) = 0.31

✅ The reward for the good prompt is higher.

---

## 🚀 Try It Yourself

```bash
pip install transformers datasets trl torch
python train_reward_model.py
