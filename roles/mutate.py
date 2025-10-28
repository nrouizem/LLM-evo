from utils.llm import resolve_model_client


def _build_mutation_prompt(
        prev: tuple[str, float],
        memory: list[tuple[str, float]]
    ) -> str:
    """
    Build the system prompt for the mutation operator so the call helper stays lean.
    """
    memory_text = "\n\n".join(
        f"--- Function (Score: {score:.4f}) ---\n{code.strip()}"
        for code,
         score in memory
    )

    return f"""
    You are performing an evolutionary **mutation**.
    Your goal is to make a small, intelligent improvement on the "Parent Solution" to increase its score.
    This improvement may add, reduce, or maintain the parent's complexity.
    The user prompt will provide further context as to the objective, the target you are mutating, and your role.

    --- Parent Solution (Score: {prev[1]:.4f}) ---
    {prev[0].strip()}

    --- Other solutions with their scores for context ---
    {memory_text}

    --- INSTRUCTIONS ---
    1.  Analyze the Parent Solution and its score.
    2.  Use the other solutions and how they scored to learn about the underlying, hidden objective.
    3.  Generate a new, improved version of the Parent.
    4.  **Write your reasoning** for this new approach, clearly
        labeled "Reasoning: {{reasoning}}". If you are modifying a function,
        place your reasoning in a comment or docstring.
    5.  **Return only the complete, new solution.** Do not add any other text before or after the solution.
    """


def call_llm_mutate(
        prev: tuple[str, float],
        query: str,
        memory: list[tuple[str, float]],
        model: str | None = None,
        client=None,
        temp: float = 0.3
    ) -> str:
    """
    Minimal example LLM call for a mutation.
    """
    sys_prompt = _build_mutation_prompt(prev, memory)
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
