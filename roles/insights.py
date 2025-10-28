from utils.llm import resolve_model_client


def _build_insight_generation_prompt(
        memory: list[tuple[str, float]],
        last_gen: list[tuple[str, float]]
    ) -> str:
    """
    Build the system prompt for the insight-generation operator so the call helper stays focused.
    """
    memory_text = "\n\n".join(
        f"--- Solution (Score: {score:.4f}) ---\n{code.strip()}"
        for code, score in memory
    )

    last_gen_text = "\n\n".join(
        f"--- Solution (Score: {score:.4f}) ---\n{code.strip()}"
        for code, score in last_gen
    )

    return f"""
    You are the Insight Agent in an evolutionary learning system.

    Your task is to **analyze performance trends** and **derive abstract, transferable insights** from
    recent generations of solutions. You must NOT reference any specific variable names or formulas
    — focus instead on *strategic and structural observations* that can guide improvement.
    Your insights will be fed to subsequent evolutionary agents to guide the process.
    Remember that the ultimate goal is to maximize the given objective.
    Use the solutions' scores to guide your recommendations; for example, if scores seem to plateau around
    some local minimum, consider recommending a sharp change to escape it.

    The user prompt will provide further context as to the objective, the target you are mutating, and your role.

    --- Sample of the Last Generation's Solutions (Good and Bad) ---
    {last_gen_text}

    --- Sample of Solutions Across All Generations ---
    {memory_text}

    --- INSTRUCTIONS ---
    1. Identify recurring patterns that correlate with higher scores (e.g., structure, complexity, stability).
    2. Note any recurring weaknesses or failure modes.
    3. Derive 3-5 high-level, general strategies that could improve future generations.
    4. Phrase them as **guidance sentences**, e.g.:
       - "Favor smoother transformations over deep nesting."
       - "Revisit linear combinations with balanced scaling."
       - "Increase structural diversity among top performers."
    4. Output a reasoning block clearly labeled "REASONING: {{reasoning}}",
       in which you describe your analysis and reasoning for the insights you choose.
    5. Finally, output a short section labeled:
       **INSIGHT SUMMARY:**
       followed by your synthesized insights.
    """


def _build_insight_application_prompt(
        memory: list[tuple[str, float]],
        insights: str
    ) -> str:
    """
    Build the system prompt for the insight-application operator so the call helper remains lean.
    """
    memory_text = "\n\n".join(
        f"--- Solution (Score: {score:.4f}) ---\n{code.strip()}"
        for code, score in memory
    )

    return f"""
    You are part of an LLM-driven evolutionary search system.
    Another analysis agent has reviewed the previous generation's performance and produced strategic insights.
    Your task is to apply those insights—together with the provided sample of past solutions—to generate
    a new candidate that performs better on the hidden objective.
    Use the insights to prioritize certain strategies or avoid specific pitfalls identified from the memory examples.
    The user prompt will provide further context as to the objective, the target you are mutating, and your role.

    --- Sample of Recent Solutions (Good and Bad) ---
    {memory_text}

    --- Analysis Insights ---
    {insights}

    --- INSTRUCTIONS ---
    1. Carefully study the patterns, structures, and ideas in the recent solutions above.
    2. Infer which design choices seem to correlate with higher or lower scores.
    3. Apply the analysis insights (provided by another agent) to guide your reasoning.
       - Integrate what worked.
       - Avoid repeating what failed.
    4. **Write your reasoning** for this new approach, clearly
        labeled "Reasoning: {{reasoning}}". If you are modifying a function,
        place your reasoning in a comment or docstring.
    5. Produce a **new and complete solution** implementing your improved solution.
    """


def call_llm_generate_insights(
        query: str,
        memory: list[tuple[str, float]],
        last_gen: list[tuple[str, float]],
        model: str | None = None,
        client=None,
    ) -> str:
    """
    Generates insights at the end of each generation.
    """
    sys_prompt = _build_insight_generation_prompt(memory, last_gen)
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
                "content": query
            }
        ]
    )

    result = completion.choices[0].message.content
    print("INSIGHTS (including reasoning, which doesn't currently get passed downstream):")
    print(result)

    # try to only return insights section
    start = result.find("INSIGHT SUMMARY")
    if start == -1:
        return result
    return result[start:]


def call_llm_insights(
        query: str,
        memory: list[tuple[str, float]],
        insights: str,
        model: str | None = None,
        client=None,
        temp: float = 0.5
    ) -> str:
    """
    Calls an LLM that implements LLM-generated insights.
    """
    sys_prompt = _build_insight_application_prompt(memory, insights)
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
