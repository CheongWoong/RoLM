INSTRUCTION_Math = {
    ### Standard prompting
    "standard": (
        "Answer the following math questions. "
        "Compute the correct numerical answer and provide your output in the following valid JSON format: "
        "```json {{\"{space}{answer_descriptor}\"{separator}\"{{numeric answer}}\"}}``` "
        "Ensure that the answer is a pure number without any symbols, units, or explanations. "
        "Do not include any additional text. "
    ),
    ### CoT (Chain of Thoughts)
    "cot": (
        "Answer the following math questions. "
        "Think step-by-step and provide a concise reasoning process that justifies your answer. "
        "Based on the reasoning, compute the correct numerical answer and provide your output in the following valid JSON format: "
        "```json {{\"{space}{explanation_descriptor}\"{separator}\"{{concise reasoning}}\", \"{space}{answer_descriptor}\"{separator}\"{{numeric answer}}\"}}``` "
        "Ensure the explanation is minimal sufficient. "
        "Ensure that the answer is a pure number without any symbols, units, or explanations. "
        "Do not include any additional text. "
    ),
    ### Reference-guided
    "reference": (
        "Answer the following math questions. "
        "Based on the provided reference, compute the correct numerical answer and provide your output in the following valid JSON format: "
        "```json {{\"{space}{answer_descriptor}\"{separator}\"{{numeric answer}}\"}}``` "
        "Ensure that the answer is a pure number without any symbols, units, or explanations. "
        "Do not include any additional text. "
    ),
}