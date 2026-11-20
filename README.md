# Retail Basket Prediction Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/tfyzt9ankk/badge.svg)](https://ask.thili.ai/b/tfyzt9ankk)

Retail analysts need to understand customer purchasing patterns. In this project, you will build a tool to compare how well an n-gram model and a small language model can predict the next item in a shopping basket. You'll train both models on simulated retail data and evaluate their accuracy to see which performs better.

## What you are building

**Goal.** You will build a Retail Basket Prediction Analyzer that compares the next-item prediction accuracy of an n-gram model and a small language model, both trained on simulated shopping basket data.

**Done when.** Your project is done when you have successfully trained both a next-item language model and an N-gram model on the same simulated retail basket data, and then evaluated and presented a comparison of their prediction performance.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Generate a synthetic dataset of shopping baskets with planted itemsets.**  
      Hint: Use the data generation technique from 'Teach a Language Model What Sells Together' to create your dataset.
- [ ] **2. Implement the n-gram model to predict next items in a basket.**  
      Hint: Apply the n-gram model implementation from 'Build a Next-Word Predictor from Scratch' to predict next items.
- [ ] **3. Pretrain a small language model (like in the drill) on the basket sequences.**  
      Hint: Train a small language model on your basket sequences, similar to the process in 'Teach a Language Model What Sells Together'.
- [ ] **4. Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).**  
      Hint: Calculate standard next-item prediction metrics like accuracy or rank for both models.
- [ ] **5. Present a comparison of the models' performance.**  
      Hint: Summarize and visualize the performance metrics to highlight differences between the models.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Experiment with different values for 'n' in your n-gram model to find an optimal configuration.
- Adjust hyperparameters or architecture of the small language model to observe performance changes.
- Analyze specific types of prediction errors made by each model to understand their strengths and weaknesses.
