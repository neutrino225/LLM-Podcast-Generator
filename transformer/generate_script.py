import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pypdf import PdfReader


def extract_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


model_folder = "./podcast_transformer_epoch_10"
tokenizer_folder = "./podcast_tokenizer_epoch_10"


def load_model(model_folder, tokenizer_folder):
    """
    Load the model and tokenizer from the given folder paths.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_folder)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder)
    return model, tokenizer


def generate_script(article, max_length=512, num_beams=4):
    """
    Generate a podcast script from a given article.
    """
    model, tokenizer = load_model(model_folder, tokenizer_folder)

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


if __name__ == "__main__":
    article = extract_text("../docs/attentionisallyouneed.pdf")
    script = generate_script(article)
    print(script)
