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
    ###########################################################
    ### Reference-guided
    "reference": (
        "Answer the following math questions. "
        "Based on the provided reference, compute the correct numerical answer and provide your output in the following valid JSON format: "
        "```json {{\"{space}{answer_descriptor}\"{separator}\"{{numeric answer}}\"}}``` "
        "Ensure that the answer is a pure number without any symbols, units, or explanations. "
        "Do not include any additional text. "
    ),
    ###########################################################
    ### guided instruction
    "guided": (
        "Answer the following math questions. "
        "Carefully read the {question_descriptor}. "
        "Compute the correct numerical answer and provide your output in the following valid JSON format: "
        "```json {{\"{space}{answer_descriptor}\"{separator}\"{{numeric answer}}\"}}``` "
        "Ensure that the answer is a pure number without any symbols, units, or explanations. "
        "Do not include any additional text. "
    ),
    "cot-guided": (
        "Answer the following math questions. "
        "Carefully read the {question_descriptor}. "
        "Think step-by-step and provide a concise reasoning process that justifies your answer. "
        "Based on the reasoning, compute the correct numerical answer and provide your output in the following valid JSON format: "
        "```json {{\"{space}{explanation_descriptor}\"{separator}\"{{concise reasoning}}\", \"{space}{answer_descriptor}\"{separator}\"{{numeric answer}}\"}}``` "
        "Ensure the explanation is minimal sufficient. "
        "Ensure that the answer is a pure number without any symbols, units, or explanations. "
        "Do not include any additional text. "
    ),
    ### components parsing
    "components": (
        "Answer the following math questions. "
        "Grammatically break down the question into its main components, such as subject, verb, object, and complement. "
        "Based on the identified components, compute the correct numerical answer and provide your output in the following valid JSON format: "
        "```json {{\"{space}components\"{separator}\"{{identified components}}\", \"{space}{answer_descriptor}\"{separator}\"{{numeric answer}}\"}}``` "
        "Ensure the breakdown is minimal sufficient. "
        "Ensure that the answer is a pure number without any symbols, units, or explanations. "
        "Do not include any additional text. "
    ),
    "cot-components": (
        "Answer the following math questions. "
        "Grammatically break down the question into its main components, such as subject, verb, object, and complement. "
        "Then, think step-by-step and provide a concise reasoning process that justifies your answer. "
        "Based on the reasoning, compute the correct numerical answer and provide your output in the following valid JSON format: "
        "```json {{\"{space}components\"{separator}\"{{identified components}}\", \"{space}{explanation_descriptor}\"{separator}\"{{concise reasoning}}\", \"{space}{answer_descriptor}\"{separator}\"{{numeric answer}}\"}}``` "
        "Ensure the breakdown and explanation are minimal sufficient. "
        "Ensure that the answer is a pure number without any symbols, units, or explanations. "
        "Do not include any additional text. "
    ),
    ### RaR (Rephrase and Respond)
    "rar": (
        "Answer the following math questions. "
        "Rephrase the question for clarity and detail. "
        "Based on the rephrased question, compute the correct numerical answer and provide your output in the following valid JSON format: "
        "```json {{\"{space}rephrased_text\"{separator}\"{{rephrased question}}\", \"{space}{answer_descriptor}\"{separator}\"{{numeric answer}}\"}}``` "
        "Ensure that the answer is a pure number without any symbols, units, or explanations. "
        "Do not include any additional text. "
    ),
    "cot-rar": (
        "Answer the following math questions. "
        "Rephrase the question for clarity and detail. "
        "Then, think step-by-step and provide a concise reasoning process that justifies your answer. "
        "Based on the reasoning, compute the correct numerical answer and provide your output in the following valid JSON format: "
        "```json {{\"{space}rephrased_text\"{separator}\"{{rephrased question}}\", \"{space}{explanation_descriptor}\"{separator}\"{{concise reasoning}}\", \"{space}{answer_descriptor}\"{separator}\"{{numeric answer}}\"}}``` "
        "Ensure the explanation is minimal sufficient. "
        "Ensure that the answer is a pure number without any symbols, units, or explanations. "
        "Do not include any additional text. "
    ),
}