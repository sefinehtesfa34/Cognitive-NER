"""
Named Entity Recognition System using BERT and Flask

This script implements a Named Entity Recognition (NER) system using a pre-trained BERT model.
It includes data preparation, model training, evaluation, and a Flask web application for inference.

Dependencies:
- transformers
- datasets
- torch
- flask
- flask-cors

To run the Flask app, use the command:
python NER.py

Make sure to activate your virtual environment before running the script.
"""

import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore

# Load the CoNLL-2003 dataset
dataset = load_dataset("conll2003")

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_and_align_labels(examples):
    """
    Tokenize the input texts and align the labels with the tokenized inputs.
    
    Args:
        examples (dict): A dictionary containing the input texts and labels.
    
    Returns:
        dict: A dictionary containing the tokenized inputs and aligned labels.
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
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

# Apply the function to the dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Load a pre-trained BERT model for token classification
model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=9)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Create a pipeline for NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

def extract_entities(text):
    """
    Extract named entities from the input text using the NER pipeline.
    
    Args:
        text (str): The input text.
    
    Returns:
        list: A list of entities with their labels and positions.
    """
    return ner_pipeline(text)

# Flask web application
app = Flask(__name__)
CORS(app)

@app.route('/extract_entities', methods=['POST'])
def extract_entities_route():
    """
    Extract entities from the input text received via POST request.
    
    Returns:
        JSON: A JSON response containing the extracted entities.
    """
    data = request.json
    text = data['text']
    entities = extract_entities(text)
    return jsonify(entities)

if __name__ == '__main__':
    app.run(debug=True)
