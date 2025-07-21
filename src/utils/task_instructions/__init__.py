from src.utils.task_instructions.MCQA import INSTRUCTION_MCQA, EINSTRUCTION_MCQA
from src.utils.task_instructions.TFQA import INSTRUCTION_TFQA
from src.utils.task_instructions.Math import INSTRUCTION_Math


INSTRUCTIONS = {
    "CommonsenseQA": INSTRUCTION_MCQA,
    "QASC": INSTRUCTION_MCQA,
    "100TFQA": INSTRUCTION_TFQA,
    "GSM8K": INSTRUCTION_Math,
    "MMLU-Pro-Law-100Q": INSTRUCTION_MCQA,
}
EINSTRUCTIONS = {
    "CommonsenseQA": EINSTRUCTION_MCQA,
    "QASC": EINSTRUCTION_MCQA,
}

PROMPTING_STRATEGY_MAP = {
    ### Standard prompting
    "zero-shot": "standard",
    "few-shot": "standard",
    ### CoT (Chain of Thoughts)
    "zero-shot-cot": "cot",
    "few-shot-cot": "cot",
    #############################################
    ### Reference-guided
    "zero-shot-reference-guided": "reference",
    #############################################
    ### guided instruction
    "zero-shot-guided": "guided",
    "few-shot-guided": "guided",
    "zero-shot-cot-guided": "cot-guided",
    "few-shot-cot-guided": "cot-guided",
    ### components parsing
    "zero-shot-components": "components",
    "few-shot-components": "components",
    "zero-shot-cot-components": "cot-components",
    "few-shot-cot-components": "cot-components",
    ### RaR (Rephrase and Respond)
    ### TODO: few-shot-rar needs rar demonstrations
    "zero-shot-rar": "rar",
    # "few-shot-rar": "rar",
    "zero-shot-cot-rar": "cot-rar",
    # "few-shot-cot-rar": "cot-rar",
}