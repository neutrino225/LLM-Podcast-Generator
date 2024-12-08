from pypdf import PdfReader
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def extract_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


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
