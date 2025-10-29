# LLM-evo

This project experiments with **LLM-driven evolution**.  
Each generation evolves a target using LLMs.

### Overview
- Combines mutation, crossover, exploration, and insight-based strategies.  
- Uses a multi-armed bandit to balance exploration and exploitation.  
- Evaluates each new gene using a downstream model or proxy objective.  
- Iteratively improves gene quality through selection and feedback.

### Goals
- Develop a **general, domain-agnostic** evolutionary framework.  
- Automate evolution across **tabular, text, or other data types**.  
- Minimize manual tuning or model-specific heuristics.  

### Current Focus
- Testing on **tabular data** (e.g., the Kaggle Calories dataset).  
- LLMs generate candidate `build_feature(df)` functions.  
- Features are scored by **correlation** or **model improvement**.  

### ⚠️ Status
- Early research prototype  
- Actively evolving and may change structure or objectives
