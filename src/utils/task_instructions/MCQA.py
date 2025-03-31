EINSTRUCTION_MCQA = {
    ### Standard prompting
    "standard": (
        "Answer the following multiple-choice questions. "
        "Select the best answer from the given options and provide your output in the following valid JSON format:\n"
        "```json {{\"{1space}{7left_bracket}{2answer_descriptor}{6descriptor_period}{7right_bracket}\"{3separator_fspace}:{4separator_bspcae}{5new_line}\"letter\"}}```\n"
        "Do not include any additional text. "
    ),
    "cot": (
        "Answer the following multiple-choice questions. "
        "Think step-by-step and provide a concise reasoning process that justifies your answer. "
        "Based on the reasoning, select the best answer from the given options and provide your output in the following valid JSON format:\n"
        "```json {{\"{1space}{7left_bracket}{2explanation_descriptor}{6descriptor_period}{7right_bracket}\"{3separator_fspace}:{4separator_bspcae}{5new_line}\"concise reasoning\", \"{1space}{7left_bracket}{2answer_descriptor}{6descriptor_period}{7right_bracket}\"{3separator_fspace}:{4separator_bspcae}{5new_line}\"letter\"}}```\n"
        "Ensure the explanation is minimal sufficient. "
        "Do not include any additional text. "
    )
}

INSTRUCTION_MCQA = {
    ### Standard prompting
    "standard": (
        "Answer the following multiple-choice questions. "
        "Select the best answer from the given options and provide your output in the following valid JSON format: "
        "```json {{\"{space}{answer_descriptor}\"{separator}\"{{letter}}\"}}``` "
        "Do not include any additional text. "
    ),
    ### CoT (Chain of Thoughts)
    "cot": (
        "Answer the following multiple-choice questions. "
        "Think step-by-step and provide a concise reasoning process that justifies your answer. "
        "Based on the reasoning, select the best answer from the given options and provide your output in the following valid JSON format: "
        "```json {{\"{space}{explanation_descriptor}\"{separator}\"{{concise reasoning}}\", \"{space}{answer_descriptor}\"{separator}\"{{letter}}\"}}``` "
        "Ensure the explanation is minimal sufficient. "
        "Do not include any additional text. "
    ),
    ###########################################################
    ### Reference-guided
    "reference": (
        "Answer the following multiple-choice questions. "
        "Based on the provided reference, select the best answer from the given options and provide your output in the following valid JSON format: "
        "```json {{\"{space}{answer_descriptor}\"{separator}\"{{letter}}\"}}``` "
        "Do not include any additional text. "
    ),
    ###########################################################
    ### guided instruction
    "guided": (
        "Answer the following multiple-choice questions. "
        "Carefully read the {question_descriptor} and {options_descriptor}. "
        "Select the best answer from the given {options_descriptor} and provide your output in the following valid JSON format: "
        "```json {{\"{space}{answer_descriptor}\"{separator}\"{{letter}}\"}}``` "
        "Do not include any additional text. "
    ),
    "cot-guided": (
        "Answer the following multiple-choice questions. "
        "Carefully read the {question_descriptor} and {options_descriptor}. "
        "Think step-by-step and provide a concise reasoning process that justifies your answer. "
        "Based on the reasoning, select the best answer from the given {options_descriptor} and provide your output in the following valid JSON format: "
        "```json {{\"{space}{explanation_descriptor}\"{separator}\"{{concise reasoning}}\", \"{space}{answer_descriptor}\"{separator}\"{{letter}}\"}}``` "
        "Ensure the explanation is minimal sufficient. "
        "Do not include any additional text. "
    ),
    ### components parsing
    "components": (
        "Answer the following multiple-choice questions. "
        "Grammatically break down the question into its main components, such as subject, verb, object, and complement. "
        "Based on the identified components, select the best answer from the given options and provide your output in the following valid JSON format: "
        "```json {{\"{space}components\"{separator}\"{{identified components}}\", \"{space}{answer_descriptor}\"{separator}\"{{letter}}\"}}``` "
        "Ensure the breakdown is minimal sufficient. "
        "Do not include any additional text. "
    ),
    "cot-components": (
        "Answer the following multiple-choice questions. "
        "Grammatically break down the question into its main components, such as subject, verb, object, and complement. "
        "Then, think step-by-step and provide a concise reasoning process that justifies your answer. "
        "Based on the reasoning, select the best answer from the given options and provide your output in the following valid JSON format: "
        "```json {{\"{space}components\"{separator}\"{{identified components}}\", \"{space}{explanation_descriptor}\"{separator}\"{{concise reasoning}}\", \"{space}{answer_descriptor}\"{separator}\"{{letter}}\"}}``` "
        "Ensure the breakdown and explanation are minimal sufficient. "
        "Do not include any additional text. "
    ),
    ### RaR (Rephrase and Respond)
    "rar": (
        "Answer the following multiple-choice questions. "
        "Rephrase the question for clarity and detail. "
        "Based on the rephrased question, select the best answer from the given options and provide your output in the following valid JSON format: "
        "```json {{\"{space}rephrased_text\"{separator}\"{{rephrased question}}\", \"{space}{answer_descriptor}\"{separator}\"{{letter}}\"}}``` "
        "Do not include any additional text. "
    ),
    "cot-rar": (
        "Answer the following multiple-choice questions. "
        "Rephrase the question for clarity and detail. "
        "Then, think step-by-step and provide a concise reasoning process that justifies your answer. "
        "Based on the reasoning, select the best answer from the given options and provide your output in the following valid JSON format: "
        "```json {{\"{space}rephrased_text\"{separator}\"{{rephrased question}}\", \"{space}{explanation_descriptor}\"{separator}\"{{concise reasoning}}\", \"{space}{answer_descriptor}\"{separator}\"{{letter}}\"}}``` "
        "Ensure the explanation is minimal sufficient. "
        "Do not include any additional text. "
    ),
}