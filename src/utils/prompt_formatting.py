from src.utils.task_instructions import INSTRUCTIONS, PROMPTING_STRATEGY_MAP, EINSTRUCTIONS
from src.utils.task_instructions.component_fewshot import components_demonstrations


final_template = "{instruction}\n\n{demonstrations}{input}"

demonstration_template = "{statement}{question}{options}```json {{{components}{explanation}{answer}}}```\n"
input_template = "{statement}{question}{options}{explanation}{answer}"

statement_template = "{space}{statement_descriptor}{separator}{statement}\n"
question_template = "{space}{question_descriptor}{separator}{question}\n"
options_template = "{space}{options_descriptor}:\n{options}\n"
fewshot_explanation_template = "\"{space}{explanation_descriptor}\"{separator}\"{explanation}\", "
fewshot_components_template = "\"{space}components\"{separator}\"{components}\", "
fewshot_answer_template = "\"{space}{answer_descriptor}\"{separator}\"{answer}\""
explanation_template = "```json {{\"{space}{explanation_descriptor}\"{separator}"
answer_template = "```json {{\"{space}{answer_descriptor}\"{separator}"
rar_template = "```json {{\"{space}rephrased_text\"{separator}"
components_template = "```json {{\"{space}components\"{separator}"
reference_template = "{space}{reference_descriptor}{separator}{reference}\n"
# fewshot_explanation_template = "{space}{explanation_descriptor}{separator}{explanation}\n"
# fewshot_answer_template = "{space}{answer_descriptor}{separator}{answer}\n"
# explanation_template = "{space}{explanation_descriptor}{separator}"
# answer_template = "{space}{answer_descriptor}{separator}"
# reference_answer_template = "{space}{reference_descriptor}{separator}{reference}\n{space}{answer_descriptor}{separator}"

DESCRIPTORS_capitalize = {
    "statement_descriptor": "Statement",
    "question_descriptor": "Question",
    "options_descriptor": "Options",
    "explanation_descriptor": "Explanation",
    "reference_descriptor": "Reference",
    "answer_descriptor": "Answer",
}
DESCRIPTORS_upper = {
    "statement_descriptor": "STATEMENT",
    "question_descriptor": "QUESTION",
    "options_descriptor": "OPTIONS",
    "explanation_descriptor": "EXPLANATION",
    "reference_descriptor": "REFERENCE",
    "answer_descriptor": "ANSWER",
}

FORMAT_TYPE_MAP = {
    "0": {"separator": ":", "casing": "capitalize", "space": ""},
    "1": {"separator": ":", "casing": "capitalize", "space": " "},
    "2": {"separator": ":", "casing": "upper", "space": ""},
    "3": {"separator": ":", "casing": "upper", "space": " "},
    "4": {"separator": ": ", "casing": "capitalize", "space": ""},
    "5": {"separator": ": ", "casing": "capitalize", "space": " "},
    "6": {"separator": ": ", "casing": "upper", "space": ""},
    "7": {"separator": ": ", "casing": "upper", "space": " "},
}

#########################################################################
REBUTTAL = True


def make_format_data(format_num):
    # descriptors
    format_data = {}
    space_1 = ("", " ")
    answer_descriptor_2 = ("Answer", "ANSWER")
    question_descriptor_2 = ("Question", "QUESTION")
    options_descriptor_2 = ("Options", "OPTIONS")
    explanation_descriptor_2 = ("Explanation", "EXPLANATION")
    separator_fspace_3 = ("", " ")
    separator_bspcae_4 = ("", " ")
    new_line_5 = ("", "\n")
    descriptor_period_6 = ("", ".")
    left_bracket_7 = ("", "<")
    right_bracket_7 = ("", ">")

    indentation_8 = ("", "\t")  # 4 spaces or \t
    list_mark_9 = ("", "* ")  
    # checking & operator
    # print(f"format_num: {format_num} -> 1={1 if format_num & 0b0000001 else 0}, 2={1 if format_num & 0b0000010 else 0}, 3={1 if format_num & 0b0000100 else 0}, 4={1 if format_num & 0b0001000 else 0}, 5={1 if format_num & 0b0010000 else 0}, 6={1 if format_num & 0b0100000 else 0}, 7={1 if format_num & 0b1000000 else 0}")


    format_data |= {"1space" : space_1[1 if format_num & 0b0000001 else 0]}                                     # Main variation 1 space before descriptor
    format_data |= {"2answer_descriptor" : answer_descriptor_2[1 if format_num & 0b0000010 else 0]}             # Main variation 2 casing of descriptor
    format_data |= {"2question_descriptor" : question_descriptor_2[1 if format_num & 0b0000010 else 0]}         # Main variation 2 casing of descriptor
    format_data |= {"2options_descriptor" : options_descriptor_2[1 if format_num & 0b0000010 else 0]}           # Main variation 2 casing of descriptor
    format_data |= {"2explanation_descriptor" : explanation_descriptor_2[1 if format_num & 0b0000010 else 0]}   # Main variation 2 casing of descriptor
    
    if not REBUTTAL:
        format_data |= {"3separator_fspace" : separator_fspace_3[1 if format_num & 0b0000100 else 0]}
    else:
        format_data |= {"3separator_fspace" : ""}            
    
    if REBUTTAL:
        format_data |= {"4separator_bspcae" : separator_bspcae_4[1 if format_num & 0b100 else 0]}               # Main variation 3 space after separator
    else:
        format_data |= {"4separator_bspcae" : separator_bspcae_4[1 if format_num & 0b0001000 else 0]}               # Main variation 3 space after separator
    
    if not REBUTTAL:
        format_data |= {"5new_line" : new_line_5[1 if format_num & 0b0010000 else 0]}
        format_data |= {"6descriptor_period" : descriptor_period_6[1 if format_num & 0b0100000 else 0]}
        format_data |= {"7left_bracket" : left_bracket_7[1 if format_num & 0b1000000 else 0]}
        format_data |= {"7right_bracket" : right_bracket_7[1 if format_num & 0b1000000 else 0]}
    else:
        format_data |= {"5new_line" : "\n"}
        format_data |= {"6descriptor_period" : ""}
        format_data |= {"7left_bracket" : ""}
        format_data |= {"7right_bracket" : ""}

    # Real-world format (for Rebuttal)
    if REBUTTAL:
        format_data |= {"8indentation": indentation_8[1 if format_num & 0b1000 else 0]}
        format_data |= {"9list_mark": list_mark_9[1 if format_num & 0b10000 else 0]}
    else:
        format_data |= {"8indentation": indentation_8[1 if format_num & 0b10000000 else 0]}
        format_data |= {"9list_mark": list_mark_9[1 if format_num & 0b100000000 else 0]}

    return format_data

def format_input_prompt_ext(input_example, prompting_strategy, format_data, is_demonstration=False):
    e_question_template = "{1space}{7left_bracket}{2question_descriptor}{6descriptor_period}{7right_bracket}{3separator_fspace}:{4separator_bspcae}{5new_line}{question}\n"
    e_options_template = "{1space}{7left_bracket}{2options_descriptor}{6descriptor_period}{7right_bracket}{3separator_fspace}:{4separator_bspcae}{5new_line}{options}\n"
    e_option_template = "{8indentation}{9list_mark}"

    e_fewshot_answer_template = "\"{1space}{2answer_descriptor}\"{3separator_fspace}:{4separator_bspcae}\"{answer}\""

    question = e_question_template.format_map(format_data | {"question": input_example["questions"]["original"]}) if "questions" in input_example else ""
    
    
    formatted_options = "\n".join([e_option_template.format_map(format_data)+f"{option['label']}. {option['text']}" for option in input_example["options"]]) if "options" in input_example else None

    options = e_options_template.format_map(format_data | {"options": formatted_options}) if "options" in input_example else ""

    if is_demonstration:
        answer = e_fewshot_answer_template.format_map(format_data | input_example)
        input_prompt = demonstration_template.format(statement="", question=question, options=options, explanation="", components="", answer=answer)
    else:
        explanation = ""
        answer = ""

        input_prompt = input_template.format(statement="", question=question, options=options, explanation=explanation, answer=answer)
    return input_prompt

def format_example_ext(example, dataset_name, prompting_strategy, format_num):
    format_data = make_format_data(format_num)

    instruction_template = EINSTRUCTIONS[dataset_name][PROMPTING_STRATEGY_MAP[prompting_strategy]]
    instruction = instruction_template.format_map(format_data)
    if "few-shot-components" in prompting_strategy:
        raise NotImplementedError
    elif "few-shot" in prompting_strategy:
        demonstrations = "\n".join([format_input_prompt_ext(demonstration_example, prompting_strategy, format_data, is_demonstration=True) for demonstration_example in example["demonstrations"]])+"\n" if "few-shot" in prompting_strategy else ""
    else:
        demonstrations = ""
        # demonstrations = "\n".join([format_input_prompt(demonstration_example, prompting_strategy, format_data, is_demonstration=True) for demonstration_example in example["demonstrations"]])+"\n" if "few-shot" in prompting_strategy else ""
    input = format_input_prompt_ext(example, prompting_strategy, format_data)
    formatted_example = final_template.format(instruction=instruction, demonstrations=demonstrations, input=input)
    return formatted_example

def format_example(example, dataset_name, prompting_strategy, format_type):
    if FORMAT_TYPE_MAP[format_type]["casing"] == "capitalize":
        descriptors = DESCRIPTORS_capitalize
    elif FORMAT_TYPE_MAP[format_type]["casing"] == "upper":
        descriptors = DESCRIPTORS_upper
    format_data = descriptors | FORMAT_TYPE_MAP[format_type]

    instruction = format_instruction_prompt(dataset_name, prompting_strategy, format_data)
    if "few-shot-components" in prompting_strategy:
        demonstrations = "\n".join([format_input_prompt(demonstration_example, prompting_strategy, format_data, is_demonstration=True) for demonstration_example in components_demonstrations[dataset_name]])+"\n"
    else:
        demonstrations = "\n".join([format_input_prompt(demonstration_example, prompting_strategy, format_data, is_demonstration=True) for demonstration_example in example["demonstrations"]])+"\n" if "few-shot" in prompting_strategy else ""
    input = format_input_prompt(example, prompting_strategy, format_data)
    formatted_example = final_template.format(instruction=instruction, demonstrations=demonstrations, input=input)
    return formatted_example

def format_instruction_prompt(dataset_name, prompting_strategy, format_data):
    instruction_template = INSTRUCTIONS[dataset_name][PROMPTING_STRATEGY_MAP[prompting_strategy]]
    instruction = instruction_template.format_map(format_data)
    return instruction

def format_input_prompt(input_example, prompting_strategy, format_data, is_demonstration=False):
    statement = statement_template.format_map(format_data | {"statement": input_example["statements"]["original"]}) if "statements" in input_example else ""
    question = question_template.format_map(format_data | {"question": input_example["questions"]["original"]}) if "questions" in input_example else ""
    options = options_template.format_map(format_data | {"options": format_options(input_example["options"])}) if "options" in input_example else ""
    if is_demonstration:
        components = fewshot_components_template.format_map(format_data | input_example) if "components" in prompting_strategy else ""
        explanation = fewshot_explanation_template.format_map(format_data | input_example) if "cot" in prompting_strategy else ""
        answer = fewshot_answer_template.format_map(format_data | input_example)
        input_prompt = demonstration_template.format(statement=statement, question=question, options=options, explanation=explanation, components=components, answer=answer)
    else:
        if "cot" in prompting_strategy:
            explanation = explanation_template.format_map(format_data | input_example)
        elif "reference-guided" in prompting_strategy:
            input_example["reference"] = input_example["fact1"] if "fact1" in input_example else input_example["explanation"]
            explanation = reference_template.format_map(format_data | input_example)
        else:
            explanation = ""

        if "components" in prompting_strategy:
            answer = components_template.format_map(format_data | input_example)
        elif "rar" in prompting_strategy:
            answer = rar_template.format_map(format_data | input_example)
        else:
            answer = answer_template.format_map(format_data | input_example) if "cot" not in prompting_strategy else ""

        # disable prefilling output format
        explanation = explanation if "reference-guided" in prompting_strategy else ""
        answer = ""

        input_prompt = input_template.format(statement=statement, question=question, options=options, explanation=explanation, answer=answer)
    return input_prompt

def format_options(options):
    formatted_options = "\n".join([f"{option['label']}. {option['text']}" for option in options])
    return formatted_options