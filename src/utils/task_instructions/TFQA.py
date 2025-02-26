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
    ###########################################################
    ### Reference-guided
    "reference": (
        "Answer the following true or false questions. "
        "Based on the provided reference, determine whether the statement is True or False and provide your output in the following valid JSON format: "
        "```json {{\"{space}{answer_descriptor}\"{separator}\"{{True/False}}\"}}``` "
        "Do not include any additional text. "
    ),
    ###########################################################
    ### guided instruction
    "guided": (
        "Answer the following true or false questions. "
        "Carefully read the {statement_descriptor}. "
        "Determine whether the {statement_descriptor} is True or False and provide your output in the following valid JSON format: "
        "```json {{\"{space}{answer_descriptor}\"{separator}\"{{True/False}}\"}}``` "
        "Do not include any additional text. "
    ),
    "cot-guided": (
        "Answer the following true or false questions. "
        "Carefully read the {statement_descriptor}. "
        "Think step-by-step and provide a concise reasoning process that justifies your answer. "
        "Based on the reasoning, determine whether the {statement_descriptor} is True or False and provide your output in the following valid JSON format: "
        "```json {{\"{space}{explanation_descriptor}\"{separator}\"{{concise reasoning}}\", \"{space}{answer_descriptor}\"{separator}\"{{True/False}}\"}}``` "
        "Ensure the explanation is minimal sufficient. "
        "Do not include any additional text. "
    ),
    ### components parsing
    "components": (
        "Answer the following true or false questions. "
        "Grammatically break down the statement into its main components, such as subject, verb, object, and complement. "
        "Based on the identified components, determine whether the statement is True or False and provide your output in the following valid JSON format: "
        "```json {{\"{space}components\"{separator}\"{{identified components}}\", \"{space}{answer_descriptor}\"{separator}\"{{True/False}}\"}}``` "
        "Ensure the breakdown is minimal sufficient. "
        "Do not include any additional text. "
    ),
    "cot-components": (
        "Answer the following true or false questions. "
        "Grammatically break down the statement into its main components, such as subject, verb, object, and complement. "
        "Then, think step-by-step and provide a concise reasoning process that justifies your answer. "
        "Based on the reasoning, determine whether the statement is True or False and provide your output in the following valid JSON format: "
        "```json {{\"{space}components\"{separator}\"{{identified components}}\", \"{space}{explanation_descriptor}\"{separator}\"{{concise reasoning}}\", \"{space}{answer_descriptor}\"{separator}\"{{True/False}}\"}}``` "
        "Ensure the breakdown and explanation are minimal sufficient. "
        "Do not include any additional text. "
    ),
    ### RaR (Rephrase and Respond)
    "rar": (
        "Answer the following true or false questions. "
        "Rephrase the statement for clarity and detail. "
        "Based on the rephrased statement, determine whether the statement is True or False and provide your output in the following valid JSON format: "
        "```json {{\"{space}rephrased_text\"{separator}\"{{rephrased statement}}\", \"{space}{answer_descriptor}\"{separator}\"{{True/False}}\"}}``` "
        "Do not include any additional text. "
    ),
    "cot-rar": (
        "Answer the following true or false questions. "
        "Rephrase the statement for clarity and detail. "
        "Then, think step-by-step and provide a concise reasoning process that justifies your answer. "
        "Based on the reasoning, determine whether the statement is True or False and provide your output in the following valid JSON format: "
        "```json {{\"{space}rephrased_text\"{separator}\"{{rephrased statement}}\", \"{space}{explanation_descriptor}\"{separator}\"{{concise reasoning}}\", \"{space}{answer_descriptor}\"{separator}\"{{True/False}}\"}}``` "
        "Ensure the explanation is minimal sufficient. "
        "Do not include any additional text. "
    ),
}