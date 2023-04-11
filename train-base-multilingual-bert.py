"""
"""
import os
import numpy as np
from datasets import DatasetDict
from transformers import (
                          BertTokenizer, 
                          BertModelWithHeads)
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction

model_name = "bert-base-multilingual-cased"
adapter_name = "nli_adapter"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModelWithHeads.from_pretrained(model_name)


def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(
      batch["query"],
      batch["answer"],
      max_length=512,
      truncation=True,
      padding="max_length"
  )

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}


dataset = DatasetDict.load_from_disk(os.path.join("data", "datasets.huggingface"))
print(dataset)

dataset = dataset.map(encode_batch, batched=True)
model.add_adapter(adapter_name)

# Add a matching classification head
model.add_classification_head(adapter_name, num_labels=2)
# Activate the adapter
model.train_adapter(adapter_name)


training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=15,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    save_steps=1000,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_accuracy,
)


trainer.train()
trainer.evaluate()

adapter_path = os.path.join(os.getcwd(), "models", "nli-adapter-bert-big")
model.save_adapter(adapter_path, adapter_name)
