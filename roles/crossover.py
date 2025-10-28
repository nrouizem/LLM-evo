from utils.llm import resolve_model_client


def _build_crossover_prompt(
        prev1: tuple[str, float],
        prev2: tuple[str, float],
        memory: list[tuple[str, float]]
    ) -> str:
    """
    Build the system prompt for the crossover operator so both the OO and functional
    entry points share identical messaging.
    """
    memory_text = "\n\n".join(
        f"--- Function (Score: {score:.4f}) ---\n{code.strip()}"
        for code, score in memory
    )

    return f"""
    You are performing an evolutionary **crossover**.
    Your goal is to combine the best ideas from both functions to create a new, hybrid function that is better than either parent.
    Identify the key components or mechanisms in each parent. Combine these components synergistically.
    The user prompt will provide further context as to the objective, the target you are mutating, and your role.

    --- Parent 1 Solution (Score: {prev1[1]:.4f}) ---
    {prev1[0].strip()}

    --- Parent 2 Solution (Score: {prev2[1]:.4f}) ---
    {prev2[0].strip()}

    --- Other solutions with their scores for context ---
    {memory_text}

    --- INSTRUCTIONS ---
    1.  Analyze the Parent Solutions and their scores.
    2.  Use the other solutions and how they scored to learn about the underlying, hidden objective.
    3.  Combine the best ideas from both solutions to create a new, hybrid solution.
    4.  **Write your reasoning** for this new approach, clearly
        labeled "Reasoning: {{reasoning}}". If you are modifying a function,
        place your reasoning in a comment or docstring.
    5.  **Return only the complete, new solution.** Do not add any other text before or after the solution.
    """

def call_llm_crossover(
        prev1: tuple[str, float],
        prev2: tuple[str, float],
        query: str,
        memory: list[tuple[str, float]],
        model: str | None = None,
        client=None,
        temp: float = 0.3
    ) -> str:
    """
    Minimal example LLM call for a crossover.
    """
    sys_prompt = _build_crossover_prompt(prev1, prev2, memory)
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
