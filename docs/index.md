#  From Functional Anlaysis to Reinforcement Learning and personalized search

Welcome to the documentation for **Reinforcement Learning for Dynamic Search Boost Optimization**.

This project combines a synthetic e-commerce search simulator with book-quality documentation exploring RL techniques for search ranking optimization.

## Quick Links

- [Book Overview](book/README.md) - Start here for the textbook content
- [Book Outline](book/outline.md) - Chapter-to-code mapping
- [Knowledge Graph](knowledge_graph/README.md) - Concept mapping and validation tools

## Book Chapters

### Part I - Foundations
- [Chapter 0: Motivation & First Experiment](book/ch00/ch00_motivation_first_experiment_revised.md)
- [Chapter 1: Search Ranking as Optimization](book/ch01/ch01_foundations_revised_math+pedagogy_v3.md)
- [Chapter 2: Probability, Measure, and Click Models](book/ch02/ch02_probability_measure_click_models.md)
- [Chapter 3: Stochastic Processes & Bellman Foundations](book/ch03/ch03_stochastic_processes_bellman_foundations.md)

### Part II - Simulator
- [Chapter 4: Generative World Design](book/ch04/ch04_generative_world_design.md)
- [Chapter 5: Relevance, Features & Reward](book/ch05/ch05_relevance_features_reward.md)

### Part III - Policies
- [Chapter 6: Discrete Template Bandits](book/ch06/discrete_template_bandits.md)
- [Chapter 6a: Neural Bandits](book/ch06a/ch06a_neural_bandits.md)
- [Chapter 7: Continuous Actions](book/ch07/ch07_continuous_actions.md)
- [Chapter 8: Policy Gradients](book/ch08/chapter08_policy_gradients_complete.md)

### Part IV - Evaluation & Deployment
- [Chapter 9: Off-Policy Evaluation](book/ch09/ch09_off_policy_evaluation.md)
- [Chapter 10: Robustness & Guardrails](book/ch10/ch10_robustness_guardrails.md)

## Getting Started

```bash
# Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate

# Install project in editable mode
python -m pip install -e .[dev]

# Run tests
pytest -q
```
