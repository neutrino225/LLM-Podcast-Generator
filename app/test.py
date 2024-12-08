import pandas as pd
from time import time

from llm.main import generate_podcast_script
from llm.rag import podcast_from_rag
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from utils.utils import extract_text

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import requests

PODCAST_SAMPLES_DIR = "./podcast_samples"

# Initialize models globally
ollama_model = None
embedding_model = None


def initialize_models():
    global ollama_model, embedding_model

    if ollama_model is None:
        print("Initializing Ollama model...")
        ollama_model = OllamaLLM(model="llama3.2:3b", base_url="http://127.0.0.1:11434")
        print("Ollama model initialized.")

    if embedding_model is None:
        print("Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Embedding model initialized.")


def calculate_rouge(predictions, references):
    """
    Calculate ROUGE scores for a list of predictions and references.

    Args:
        predictions (list of str): List of generated texts.
        references (list of str): List of reference texts.

    Returns:
        dict: Average ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1, rouge2, rougeL = 0, 0, 0

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1 += scores["rouge1"].fmeasure
        rouge2 += scores["rouge2"].fmeasure
        rougeL += scores["rougeL"].fmeasure

    n = len(predictions)
    return {"rouge1": rouge1 / n, "rouge2": rouge2 / n, "rougeL": rougeL / n}


def calculate_bleu(predictions, references):
    """
    Calculate BLEU scores for a list of predictions and references.

    Args:
        predictions (list of str): List of generated texts.
        references (list of str): List of reference texts.

    Returns:
        float: Average BLEU score.
    """
    smoothie = SmoothingFunction().method4
    bleu_score = 0

    for pred, ref in zip(predictions, references):
        # BLEU expects references to be a list of tokens
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()
        bleu_score += sentence_bleu(
            ref_tokens, pred_tokens, smoothing_function=smoothie
        )

    return bleu_score / len(predictions)


def test():
    df = pd.read_csv("./podcast_samples/podcast_db.csv")

    inference_times = {
        "Base LLM": [],
        "RAG": [],
        "Custom LLM": [],
    }

    rouge_scores_each = {
        "Base LLM": [],
        "RAG": [],
        "Custom LLM": [],
    }

    bleu_scores_each = {
        "Base LLM": [],
        "RAG": [],
        "Custom LLM": [],
    }

    for i, row in df.iterrows():
        pdf_file = row["file"]
        podcast_file = row["podcast"]

        with open(podcast_file, "r") as f:
            podcast_script = f.read()

        ## Base LLM
        time_start = time()
        script = generate_podcast_script(model=ollama_model, file_path=pdf_file)
        time_end = time()
        inference_times["Base LLM"].append(time_end - time_start)

        ## calculate rouge and bleu scores
        rouge_scores = calculate_rouge([script], [podcast_script])
        bleu_score = calculate_bleu([script], [podcast_script])

        rouge_scores_each["Base LLM"].append(rouge_scores)
        bleu_scores_each["Base LLM"].append(bleu_score)

        ## RAG
        time_start = time()
        script = podcast_from_rag(
            model=ollama_model, embedding_model=embedding_model, pdf_path=pdf_file
        )
        time_end = time()
        inference_times["RAG"].append(time_end - time_start)

        ## calculate rouge and bleu scores
        rouge_scores = calculate_rouge([script], [podcast_script])
        bleu_score = calculate_bleu([script], [podcast_script])

        rouge_scores_each["RAG"].append(rouge_scores)
        bleu_scores_each["RAG"].append(bleu_score)

        ## Custom LLM
        time_start = time()
        article = extract_text(pdf_file)
        url = "http://localhost:5000/generate-script"
        data = {"text": article}
        response = requests.post(url, json=data)
        response_data = response.json()
        script = response_data["script"]
        time_end = time()
        inference_times["Custom LLM"].append(time_end - time_start)

        ## calculate rouge and bleu scores
        rouge_scores = calculate_rouge([script], [podcast_script])
        bleu_score = calculate_bleu([script], [podcast_script])

        rouge_scores_each["Custom LLM"].append(rouge_scores)
        bleu_scores_each["Custom LLM"].append(bleu_score)

        break

    print(f"------ Base LLM ------")
    print(
        f"Average inference time: {sum(inference_times['Base LLM']) / len(inference_times['Base LLM'])}"
    )
    print(
        f"Average ROUGE-1: {sum([r['rouge1'] for r in rouge_scores_each['Base LLM']]) / len(rouge_scores_each['Base LLM'])}"
    )
    print(
        f"Average ROUGE-2: {sum([r['rouge2'] for r in rouge_scores_each['Base LLM']]) / len(rouge_scores_each['Base LLM'])}"
    )
    print(
        f"Average ROUGE-L: {sum([r['rougeL'] for r in rouge_scores_each['Base LLM']]) / len(rouge_scores_each['Base LLM'])}"
    )
    print(
        f"Average BLEU: {sum(bleu_scores_each['Base LLM']) / len(bleu_scores_each['Base LLM'])}"
    )

    print(f"------ RAG ------")
    print(
        f"Average inference time: {sum(inference_times['RAG']) / len(inference_times['RAG'])}"
    )
    print(
        f"Average ROUGE-1: {sum([r['rouge1'] for r in rouge_scores_each['RAG']]) / len(rouge_scores_each['RAG'])}"
    )
    print(
        f"Average ROUGE-2: {sum([r['rouge2'] for r in rouge_scores_each['RAG']]) / len(rouge_scores_each['RAG'])}"
    )
    print(
        f"Average ROUGE-L: {sum([r['rougeL'] for r in rouge_scores_each['RAG']]) / len(rouge_scores_each['RAG'])}"
    )
    print(
        f"Average BLEU: {sum(bleu_scores_each['RAG']) / len(bleu_scores_each['RAG'])}"
    )

    print(f"------ Custom LLM ------")
    print(
        f"Average inference time: {sum(inference_times['Custom LLM']) / len(inference_times['Custom LLM'])}"
    )
    print(
        f"Average ROUGE-1: {sum([r['rouge1'] for r in rouge_scores_each['Custom LLM']]) / len(rouge_scores_each['Custom LLM'])}"
    )
    print(
        f"Average ROUGE-2: {sum([r['rouge2'] for r in rouge_scores_each['Custom LLM']]) / len(rouge_scores_each['Custom LLM'])}"
    )
    print(
        f"Average ROUGE-L: {sum([r['rougeL'] for r in rouge_scores_each['Custom LLM']]) / len(rouge_scores_each['Custom LLM'])}"
    )
    print(
        f"Average BLEU: {sum(bleu_scores_each['Custom LLM']) / len(bleu_scores_each['Custom LLM'])}"
    )


if __name__ == "__main__":
    initialize_models()
    test()
