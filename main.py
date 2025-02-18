# Step 1: Import Libraries
import os
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset as HFDataset
import pandas as pd

# Step 2: Data Preparation
dataset = pd.read_csv('modified_dataset.csv')

# Assign X and Y
X = list(dataset['preprocessed_text'])
y = list(dataset['summary'])

# Select Specific Range of Text Data Length
X_temp, y_temp = [], []
for i in range(len(X)):
    if len(X[i]) <= 100000:
        X_temp.append(X[i])
        y_temp.append(y[i])

X, y = X_temp, y_temp

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Load Pre-trained Model and Tokenizer
model_name = "facebook/bart-base"  # Use smaller model
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Step 4: Tokenize Data
# Use smaller max_length to reduce memory usage
max_length = 256
train_encodings = tokenizer(
    X_train, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt"
)
test_encodings = tokenizer(
    X_test, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt"
)

# Tokenize target summaries as labels
train_labels = tokenizer(
    y_train, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt"
)["input_ids"]
test_labels = tokenizer(
    y_test, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt"
)["input_ids"]

# Step 5: Convert to Hugging Face Dataset
train_dataset = HFDataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})
eval_dataset = HFDataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels
})

# Step 6: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=1,  # Reduce batch size
    per_device_eval_batch_size=1,   # Reduce batch size
    gradient_accumulation_steps=8,  # Simulate larger batch size
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=1000,
    run_name="Legal_AI",
    report_to="none",  # Disable logging
    fp16=True,  # Enable mixed precision training
    save_strategy="epoch",  # Save model after each epoch
)

# Clear GPU Cache
torch.cuda.empty_cache()

# Step 7: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Clear GPU stats before training
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()

# Step 8: Train the Model
trainer.train()

# Step 9: Evaluate the Model
evaluation_results = trainer.evaluate(eval_dataset)
print("Evaluation Results:", evaluation_results)
