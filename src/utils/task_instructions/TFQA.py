INSTRUCTION_TFQA = {
    ### Standard prompting
    "standard": (
        "Answer the following true or false questions. "
        "Determine whether the statement is True or False and provide your output in the following valid JSON format: "
        "```json {{\"{space}{answer_descriptor}\"{separator}\"{{True/False}}\"}}``` "
        "Do not include any additional text. "
    ),
    ### CoT (Chain of Thoughts)
    "cot": (
        "Answer the following true or false questions. "
        "Think step-by-step and provide a concise reasoning process that justifies your answer. "
        "Based on the reasoning, determine whether the statement is True or False and provide your output in the following valid JSON format: "
        "```json {{\"{space}{explanation_descriptor}\"{separator}\"{{concise reasoning}}\", \"{space}{answer_descriptor}\"{separator}\"{{True/False}}\"}}``` "
        "Ensure the explanation is minimal sufficient. "
        "Do not include any additional text. "
    ),
    ### Reference-guided
    "reference": (
        "Answer the following true or false questions. "
        "Based on the provided reference, determine whether the statement is True or False and provide your output in the following valid JSON format: "
        "```json {{\"{space}{answer_descriptor}\"{separator}\"{{True/False}}\"}}``` "
        "Do not include any additional text. "
    ),
}