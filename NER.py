import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset, load_metric
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore
from flask import render_template

# Load dataset
dataset = load_dataset("conll2003")
label_list = dataset['train'].features['ner_tags'].feature.names

# Initialize tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list))

app = Flask(__name__)
CORS(app)

@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')

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
