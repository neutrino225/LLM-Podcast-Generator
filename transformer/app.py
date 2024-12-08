from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pypdf import PdfReader

app = Flask(__name__)

# Load the model and tokenizer (preloaded for better performance)
MODEL_FOLDER = "./podcast_transformer_epoch_10"
TOKENIZER_FOLDER = "./podcast_tokenizer_epoch_10"

print("Loading model")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_FOLDER)
print("Model loaded\n")

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FOLDER)

print("Tokenizer loaded")


def extract_text(file_path):
    """
    Extract text from a PDF file.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def generate_script(article, max_length=512, num_beams=4):
    """
    Generate a podcast script from a given article.
    """
    # Tokenize the input article
    inputs = tokenizer(
        article,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Move inputs to the device
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Generate the output sequence
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    # Decode the generated script
    script = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return script


@app.route("/generate-script", methods=["POST"])
def generate_script_endpoint():
    """
    Flask endpoint to generate a podcast script.
    """
    try:
        # Read the input data
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Invalid input. 'text' field is required."}), 400

        article = data["text"]

        # Generate the script
        script = generate_script(article)
        return jsonify({"script": script})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000)
