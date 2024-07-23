import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd


tokenizer = AutoTokenizer.from_pretrained("PartAI/Dorna-Llama3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "amir-ma71/Dorna-Llama3-8B-Instruct-AWQ",
    torch_dtype=torch.float16,
    device_map="cuda",
)

# messages = [
#     {"role": "system",
#      "content": "You are a helpful Persian assistant. Please answer questions in the asked language."},
#     {"role": "user", "content": "پایتخت ایران کجاست؟"},
# ]



def dornai(chat:str):
    try:
        messages = [
            {"role": "system",
             "content": "You are a helpful Persian assistant. Please answer questions in the asked language."},#"You are a helpful Persian assistant."},
            {"role": "user", "content": chat},
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
        # print(tokenizer.decode(response, skip_special_tokens=True))
        return "Accomplished",tokenizer.decode(response, skip_special_tokens=True)
    except Exception as e:
        return str(e),"قادر به پاسخگویی برای متنی که فرستادید نیستم."


# a = pd.read_csv("/home/makhataei/Projects/chatbot/src/QAs.csv", on_bad_lines="skip")
# for q in a.question:
#     b=dornai(q)
#     # print(b[1])
#     with open('tesr.csv', 'a') as file:
#         file.writelines(
#             f'{q},{b[1]}\n')

# print("yaya")