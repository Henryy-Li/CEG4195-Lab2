'''
Course: CEG 4195
Name:   Henry Li

Instructions for Running the Container Locally:
See lab document.
'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from flask import Flask, request, jsonify       
import torch.nn.functional as Functions  

# ===== Create the Flask app =====
app = Flask(__name__)

# ===== Retrieve model tokenizer and model itself =====
modelName = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModelForSequenceClassification.from_pretrained(modelName)
model.eval()    #Switch model from training to evaluation mode.

# ===== Application =====
@app.route("/predict", methods=["POST"])

def predict():                      #Function that does the sentiment analysis
    # === Process the input ===
    JSONData = request.json
    textData = JSONData["text"]
    modelInputs = tokenizer(textData, return_tensors="pt")

    # === Model output and processing ===
    modelOutputs = model(**modelInputs)
    modelPredictions = Functions.softmax(modelOutputs.logits, dim=-1)      #Take raw scores and turn them into probabilities.

    # === Create the JSON output ===
    outputLabel = model.config.id2label[torch.argmax(modelPredictions).item()]          # Positive or negative setiment.
    outputScore = modelPredictions[0][torch.argmax(modelPredictions)].item()            # Percetage that it is that label classification.
    return jsonify({"label": outputLabel, "score": outputScore})

if __name__ == "__main__":          # App runs only if the script is ran directly.
    app.run(debug=True)