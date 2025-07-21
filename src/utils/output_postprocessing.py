from functools import partial
import re

DEBUG = False

#########################################################################

MCQA_OPTIONS = [chr(65 + i) for i in range(10)]

def validate_output_TFQA(output):
    is_valid = output in ["True", "False"]
    return is_valid

def validate_output_MCQA(output, num_options):
    is_valid = output in MCQA_OPTIONS[:num_options]
    return is_valid

def validate_output_Math(output):
    try:
        float(output)
        is_valid = True
    except:
        is_valid = False
    return is_valid

#########################################################################

def parse_output(raw_output, dataset_name):
    if dataset_name in ["100TFQA"]:
        return parse_output_TFQA(raw_output)
    elif dataset_name in ["CommonsenseQA"]:
        return parse_output_MCQA(raw_output, num_options=5)
    elif dataset_name in ["QASC"]:
        return parse_output_MCQA(raw_output, num_options=8)
    elif dataset_name in ["GSM8K"]:
        return parse_output_Math(raw_output)
    elif dataset_name in ["MMLU-Pro-Law-100Q"]:
        return parse_output_MCQA(raw_output, num_options=10)
    else:
        raise NotImplementedError

def parse_output_TFQA(raw_output):
    pattern = "\s*[Aa][Nn][Ss][Ww][Ee][Rr][Ss]?\s*\"?\s*:\s*\"?([Tt][Rr][Uu][Ee]|[Ff][Aa][Ll][Ss][Ee])"
    match = re.search(pattern, raw_output)
    output = match.group(1).capitalize() if match else "None"
    if validate_output_TFQA(output):
        return output, True
    
    pattern = ":\s*\"?([Tt][Rr][Uu][Ee]|[Ff][Aa][Ll][Ss][Ee])"
    match = re.search(pattern, raw_output)
    output = match.group(1).capitalize() if match else "None"
    if validate_output_TFQA(output):
        return output, True
    
    pattern = ":\s*\"{?([Tt][Rr][Uu][Ee]|[Ff][Aa][Ll][Ss][Ee])"
    match = re.search(pattern, raw_output)
    output = match.group(1).capitalize() if match else "None"
    if validate_output_TFQA(output):
        return output, True
    else:
        if DEBUG:
            print('='*20)
            print("PARSING FAILED!", output, raw_output)
            print('='*20)
        return output, False

def parse_output_MCQA(raw_output, num_options):
    pattern = "\s*[Aa][Nn][Ss][Ww][Ee][Rr][Ss]?\s*\"?\s*:\s*\"?([A-J])"
    match = re.search(pattern, raw_output)
    output = match.group(1) if match else "None"
    if validate_output_MCQA(output, num_options):
        return output, True

    pattern = "\s*[Aa]?\s*\"?\s*:\s*\"?([A-J])"
    match = re.search(pattern, raw_output)
    output = match.group(1) if match else "None"
    if validate_output_MCQA(output, num_options):
        return output, True

    pattern = "\s*[Aa]?\s*\"?\s*:\s*\"{?([A-J])"
    match = re.search(pattern, raw_output)
    output = match.group(1) if match else "None"
    if validate_output_MCQA(output, num_options):
        return output, True

    pattern = "\s*[Aa]?\s*\"?\s*:\s*{\"?([A-J])"
    match = re.search(pattern, raw_output)
    output = match.group(1) if match else "None"
    if validate_output_MCQA(output, num_options):
        return output, True
    else:
        if DEBUG:
            print('='*20)
            print("PARSING FAILED!", output, raw_output)
            print('='*20)
        return output, False

def parse_output_Math(raw_output):
    pattern = "\s*[Aa][Nn][Ss][Ww][Ee][Rr][Ss]?\s*\"?\s*:\s*\"?([^\"?}]*)"
    match = re.search(pattern, raw_output)
    output = match.group(1) if match else "None"
    try:
        output = output.replace(",", "").replace("^", "**").replace(":", "/")
        output = eval(output)
    except:
        numbers = re.findall(r'-?\d+\.?\d*', output)
        if numbers:
            final_number = numbers[-1]
            output = float(final_number)
    if validate_output_Math(output):
        # numeric output normalization
        output = round(float(output), 2)
        output = int(output) if output.is_integer() else output
        return str(output), True
    else:
        if DEBUG:
            print('='*20)
            print("PARSING FAILED!", output, raw_output)
            print('='*20)
        return str(output), False