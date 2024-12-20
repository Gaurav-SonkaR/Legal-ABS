# from transformers import TFTrainer
# from transformers import TFTrainingArguments
import tensorflow as tf
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from transformers import BartForConditionalGeneration, BartTokenizer


# load the dataset
dataset = pd.read_csv('modified_dataset.csv')
print(dataset.info())

# Make x and y
X = list(dataset['preprocessed_text'])
y = list(dataset['summary'])

# find no. of rows
len_list = []
for x in X:
    len_list.append(len(x))
print(len(len_list))

# use only small dataset
X_temp = []
y_temp = []
for i in range(len(X)):
    if len(X[i]) <=10000:
        X_temp.append(X[i])
        y_temp.append(y[i])

print(len(X_temp))

X = X_temp
y = y_temp


# split the dataset
X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("split is done")

print("model is start")
model_name = "facebook/bart-large-cnn"

# Load model and tokenizer with no progress bar
model = BartForConditionalGeneration.from_pretrained(model_name, use_progress_bar=False)
tokenizer = BartTokenizer.from_pretrained(model_name, use_progress_bar=False)

print("Model and tokenizer loaded successfully!")

print("tokenizer  is start")
train_encoding = tokenizer(X_train,max_length=1024, truncation=True, padding="max_length" )
test_encoding = tokenizer(X_test ,max_length= 1024, truncation=True, padding="max_length")


print("dataset construction is start")
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encoding),
    y_train
))


test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encoding),
    y_test
))


print("training arg is start")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,  # Adjust as needed
    per_device_train_batch_size=4,
    per_device_eval_batch_size= 4,
    warmup_steps=500,
    weight_decay = 0.01,
    logging_dir="./logs",
    logging_steps=1000,

    # save_steps=10_000,
    # save_total_limit=2,
    # evaluation_strategy="epoch",
)

print("training is start")

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset
)

trainer.train()


