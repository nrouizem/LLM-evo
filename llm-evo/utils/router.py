import numpy as np
from collections import defaultdict

from roles.crossover import call_llm_crossover
from roles.explore import call_llm_explore
from roles.mutate import call_llm_mutate
from roles.insights import call_llm_insights

def operator_router(prev1, prev2, query, memory, bandit, insights):
    op = bandit.pick()
    if op == "explore":
        mutation = call_llm_explore(query, memory)
    elif op == "cross":
        mutation = call_llm_crossover(prev1, prev2, query, memory)
    elif op == "insights":
        mutation = call_llm_insights(query, memory, insights)
    elif op == "mutate":
        mutation = call_llm_mutate(prev1, query, memory)
    else:
        raise NotImplementedError("This op isn't implemented yet.")
    return op, mutation, prev1[1], prev2[1]

class OpBandit:
    def __init__(self, ops=("mutate","cross","explore","insights"), init_probs=None):
        self.ops = ops
        self.R = defaultdict(lambda: 0.0)
        self.N = defaultdict(lambda: 1e-6)
        if init_probs is None:
            init_probs = np.ones(len(ops))/len(ops)
        init_probs = np.array(init_probs)/np.sum(init_probs)
        self.priors = dict(zip(ops, init_probs))
        self.probs = init_probs.copy()

    def pick(self, tau=0.3):
        avgs = np.array([self.R[o]/self.N[o] for o in self.ops])
        # temperature only on reward term; prior added unscaled
        logits = (avgs / max(tau, 1e-8)) + np.log(np.array([self.priors[o] for o in self.ops]) + 1e-8)
        probs = np.exp(logits)
        self.probs = probs / probs.sum()
        return np.random.choice(self.ops, p=self.probs)

    def update(self, op, reward):
        self.R[op] += reward; self.N[op] += 1

    def __repr__(self):
        return "\n".join([f"{op} probability: {self.probs[i]:.4f}" for i, op in enumerate(self.ops)])