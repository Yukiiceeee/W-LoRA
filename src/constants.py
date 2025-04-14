PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "prompt_mc_input": (
        "Below is a multiple choice question. Write a response that contains the letter of the correct answer followed by its complete content (e.g., 'B. Head is the right part').\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_mc_no_input": (
        "Below is a multiple choice question. Write a response that contains the letter of the correct answer followed by its complete content (e.g., 'B. Head is the right part').\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "prompt_ha_input": (
        "Below is a Yes/No question about a news headline. Please answer with either 'Yes' or 'No'.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_ha_no_input": (
        "Below is a Yes/No question about a news headline. Please answer with either 'Yes' or 'No'.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}

SELECT_PROMPT_START = (
    "Determine which of the following datapoint is in which domain, what is the task.\n"
    "instruction:{instruction}\n"
    "input:{input}\n"
    "output:{output}\n"
)
SELECT_PROMPT_START_TRAIN = (
    "Determine which of the following instruction-input-output datapoint is in which domain, what is the task, and what skills are required.\n"
    "instruction:{instruction}\n"
    "input:{input}\n"
)
SELECT_PROMPT_END = (
    "Select from below list:\n"
    "Domains:{domain_list}\n"
    "Tasks:{task_list}\n"
    "Results:"
)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"