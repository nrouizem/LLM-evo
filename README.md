# LLM-evo

This project experiments with **LLM-driven evolution**.  
Each generation evolves a target using LLMs.
Take a look at [this example](examples/func_opt/) which optimizes a complex operation on large NumPy arrays.

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
- Testing on **optimizing NumPy operations on large arrays**
  - LLMs generate candidate `build_func(x)` functions
  - Functions are scored by **correctness** and **performance**
  - Single-threaded function performance increased by **27%** at large `n`
- Testing on **tabular data** (e.g., the Kaggle Calories dataset)
  - LLMs generate candidate `build_feature(df)` functions
  - Features are scored by **correlation** or **model improvement**
  - Features improved downstream model performance by **5%+**

### ⚠️ Status
- Early prototype  
- Actively evolving and may change structure or objectives
