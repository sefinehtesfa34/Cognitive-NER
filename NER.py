import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset, load_metric
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load dataset
dataset = load_dataset("conll2003")
label_list = dataset['train'].features['ner_tags'].feature.names

# Initialize tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list))

# Define label_all_tokens
label_all_tokens = True

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, padding='max_length', max_length=128)
    labels = []
    for i, label in enumerate(examples[f'ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize and align labels for train and validation sets
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Load metric
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=0.01,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Data collator for padding
data_collator = DataCollatorForTokenClassification(tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Save model
trainer.save_model("ner_model")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/extract_entities', methods=['POST'])
def extract_entities():
    data = request.get_json()
    text = data['text']
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model(inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    entities = []
    for token, label_id in zip(tokens, predictions[0].tolist()):
        entities.append({"word": token, "entity": label_list[label_id]})
    return jsonify(entities)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
