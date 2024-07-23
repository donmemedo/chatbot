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

# model_checkpoint = "makhataei/qa-persian-bert-fa-base-uncased"
model_checkpoint = "makhataei/qa-persian-roberta-fa-zwnj-base"
max_length = 512  # The maximum length of a feature (question and context)
doc_stride = 256  # The authorized overlap between two part of the context when splitting it is needed.
batch_size = 24
lr = 1e-5
epoch = 10


def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = {"answer_start":[examples["answer_start"][sample_index]],"answer_text":[examples["answer_text"][sample_index]]}
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            if not start_char:
                print("zzzz")
            end_char = start_char + len(answers["answer_text"][0])
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
# my_dataset = load_dataset("Gholamreza/pquad")
# my_dataset = load_dataset("SajjadAyoubi/persian_qa")
my_dataset = load_dataset(path="/home/makhataei/Projects1/ChatBot/src")
train_dataset = my_dataset["train"]
val_dataset = my_dataset["train"]
# val_dataset = my_dataset["validation"]
# val_dataset = my_dataset["test"]


for i in range(10):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    token_train = train_dataset.map(
        prepare_train_features, batched=True, remove_columns=train_dataset.column_names
    )
    token_val = val_dataset.map(prepare_train_features, batched=True,remove_columns=train_dataset.column_names)

    args = TrainingArguments(
        # f"qa-persian-bert-fa-base-uncased",
        output_dir="/media/makhataei/Backups/qa-persian-roberta-fa-zwnj-base",
        evaluation_strategy="epoch",
        report_to=["tensorboard"],
        learning_rate=lr,
        per_device_train_batch_size=14,
        per_device_eval_batch_size=14,
        logging_strategy="epoch",
        push_to_hub=True,
        num_train_epochs=epoch,
        # auto_find_batch_size=True,
        weight_decay=0.0001,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=token_train,
        eval_dataset=token_val,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.create_model_card()
    trainer.push_to_hub()
    tokenizer.push_to_hub("makhataei/qa-persian-roberta-fa-zwnj-base")
    lr = lr / 2


# token_train = train_dataset.map(
#     prepare_train_features, batched=True, remove_columns=train_dataset.column_names
# )
# token_val = val_dataset.map(prepare_train_features, batched=True)
