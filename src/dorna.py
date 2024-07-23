from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer
from huggingface_hub import interpreter_login, login

# interpreter_login()
login(
    token="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",
    add_to_git_credential=True,
    new_session=True,
    write_permission=True,
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Use a pipeline as a high-level helper
from transformers import pipeline

# pipe = pipeline("text-generation", model="amir-ma71/Dorna-Llama3-8B-Instruct-AWQ")
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("PartAI/Dorna-Llama3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("amir-ma71/Dorna-Llama3-8B-Instruct-AWQ")


tokenizer = AutoTokenizer.from_pretrained("PartAI/Dorna-Llama3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "amir-ma71/Dorna-Llama3-8B-Instruct-AWQ",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system",
     "content": "You are a helpful Persian assistant. Please answer questions in the asked language."},
    {"role": "user", "content": "پایتخت ایران کجاست؟"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
