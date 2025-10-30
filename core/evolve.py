import json
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from utils.router import OpBandit, operator_router
from utils.client import client
from roles.insights import call_llm_generate_insights

def evolve(
        query: str,
        objective,
        seeds,
        *objective_args,
        K: int = 8,
        C: int = 2,
        GENS: int = 20,
        log_path: str | None = None
    ):
    """
    Evolution engine.
    """
    log_path = Path(log_path or "evolve_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_entry(entry_type, generation, content, score, op=None):
        record = {
            "type": entry_type,
            "generation": generation,
            "score": score,
            "text": str(content)
        }
        if op is not None:
            record["operator"] = op
        with open(log_path, "a", encoding="utf-8") as log_file:
            json.dump(record, log_file)
            log_file.write("\n")
            log_file.flush()

    llm_workers  = min(12, K*C)
    eval_workers = 1
    parallel_eval = eval_workers > 1
    bandit = OpBandit(init_probs=[0.5, 0.1, 0.3, 0.1])
    
    # 0) initial seed evals (use eval pool)
    topK = []
    if parallel_eval:
        with ProcessPoolExecutor(max_workers=eval_workers) as pool_eval:
            fut2seed = {pool_eval.submit(objective, s, *objective_args): s for s in seeds}
            print("submitted")
            for fut in as_completed(fut2seed):
                s = fut2seed[fut]
                try:
                    score = fut.result()
                    topK.append((s, score))
                    print("added seed:")
                    print(s)
                except Exception:
                    score = 0.0
                    topK.append((s, score))
                log_entry("seed", 0, s, score)
    else:
        for s in seeds:
            try:
                score = objective(s, *objective_args)
                topK.append((s, score))
                print("added seed:")
                print(s)
            except Exception:
                score = 0.0
                topK.append((s, score))
            log_entry("seed", 0, s, score)

    memory  = random.sample(topK, k=min(1, len(topK)))
    topK    = sorted(topK, key=lambda x: x[1], reverse=True)[:K]
    insights = None

    for gen in range(1, GENS+1):
        print(f"\nGENERATION {gen}\n")
        if not topK: break

        new_children = []
        if parallel_eval:
            with ThreadPoolExecutor(max_workers=llm_workers) as pool_llm, \
                ProcessPoolExecutor(max_workers=eval_workers) as pool_eval:

                # 1) submit LLM generation tasks
                gen_futs = []
                for p in topK:
                    for _ in range(C):
                        q = random.choice(topK)
                        gen_futs.append(
                            pool_llm.submit(
                                operator_router,
                                prev1=p, prev2=q, query=query,
                                memory=memory, bandit=bandit, insights=insights
                            )
                        )

                # 2) as each child is generated, submit eval immediately to eval pool
                eval_futs = {}
                for gfut in as_completed(gen_futs):
                    try:
                        op, mutation, p1s, p2s = gfut.result()
                        efut = pool_eval.submit(objective, mutation, *objective_args)
                        eval_futs[efut] = (op, mutation, p1s, p2s)
                    except Exception as e:
                        print(f"generation error: {e}")

                # 3) consume evals
                for efut in as_completed(eval_futs):
                    op, mutation, p1s, p2s = eval_futs[efut]
                    try:
                        score = efut.result()
                    except Exception as e:
                        print(f"eval error: {e}")
                        score = 0.0
                    reward = score - max(p1s, p2s) if p1s is not None and p2s is not None else score
                    bandit.update(op, reward)
                    new_children.append((mutation, score))
                    log_entry("mutation", gen, mutation, score, op=op)
                    print(f"[{op}] reward={reward:+.4f}  score={score:.4f}")
        
        else:
            with ThreadPoolExecutor(max_workers=llm_workers) as pool_llm:

                # 1) submit LLM generation tasks
                gen_futs = []
                for p in topK:
                    for _ in range(C):
                        q = random.choice(topK)
                        gen_futs.append(
                            pool_llm.submit(
                                operator_router,
                                prev1=p, prev2=q, query=query,
                                memory=memory, bandit=bandit, insights=insights
                            )
                        )

                # 2) sequentially evaluate each completed child
                for gfut in as_completed(gen_futs):
                    try:
                        op, mutation, p1s, p2s = gfut.result()
                    except Exception as e:
                        print(f"generation error: {e}")
                        continue

                    try:
                        score = objective(mutation, *objective_args)
                    except Exception as e:
                        print(f"eval error: {e}")
                        score = 0.0

                    reward = score - max(p1s, p2s) if p1s is not None and p2s is not None else score
                    bandit.update(op, reward)
                    new_children.append((mutation, score))
                    log_entry("mutation", gen, mutation, score, op=op)
                    print(f"[{op}] reward={reward:+.4f}  score={score:.4f}")
                    
        # 4) selection + memory + insights (unchanged)
        full = topK + new_children
        sorted_topK = sorted(full, key=lambda x: x[1], reverse=True)
        pop_size = len(sorted_topK)
        if pop_size >= 7:
            memory = sorted_topK[:3] + sorted_topK[-2:] + random.sample(sorted_topK[3:-2], k=2)
        elif pop_size > 0:
            memory = sorted_topK[:min(pop_size, 3)] + sorted_topK[max(-pop_size, -2):]
        else:
            memory = []
        topK = sorted_topK[:K]
        print(bandit)
        if gen != GENS:
            insights = call_llm_generate_insights(query, memory, new_children, model="gpt-5")

    return topK
