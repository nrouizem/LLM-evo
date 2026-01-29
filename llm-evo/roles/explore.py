from utils.llm import resolve_model_client


def _build_explore_prompt(
        memory: list[tuple[str, float]]
    ) -> str:
    """
    Build the system prompt for the exploration operator so the call helper stays DRY.
    """
    memory_text = "\n\n".join(
        f"--- Solution (Score: {score:.4f}) ---\n{code.strip()}"
        for code, score in memory
    )

    return f"""
    You are performing an evolutionary **exploration**.
    Your single-minded goal is to **generate diversity**.
    You must create a solution with a **fundamentally different approach**
    than the solutions provided below.
    The user prompt will provide further context as to the objective, the target you are mutating, and your role.

    --- Sample of Recent Solutions (Good and Bad) ---
    {memory_text}
    
    --- INSTRUCTIONS ---
    1.  Analyze all the solutions above. Identify the **common strategies,
        structures, or assumptions** they share.
    2.  Your task is to **deliberately break from that shared pattern.**
        Do not make a small change or improvement.
    3.  Generate a solution based on a **completely new idea** or a
        different paradigm. Think: "What is an alternative,
        untested way to approach this?"
    4.  **Write your reasoning** for this new approach, clearly
        labeled "Reasoning: {{reasoning}}". If you are modifying a function,
        place your reasoning in a comment or docstring.
    5.  **Return only the complete, new solution** in the format
        defined by the system prompt.
    """


def call_llm_explore(
        query: str,
        memory: list[tuple[str, float]],
        model: str | None = None,
        client=None,
        temp: float = 0.9
    ) -> str:
    """
    Calls an LLM to generate a brand new solution for the sake of diversity.
    This operator is domain-agnostic and focuses on being different.
    """
    sys_prompt = _build_explore_prompt(memory)

    model_name, resolved_client = resolve_model_client(model, client)

    completion = resolved_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": query,
                "temperature": temp
            }
        ]
    )

    return completion.choices[0].message.content
