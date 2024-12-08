import gradio as gr
import requests
from time import time
import pandas as pd

## Import necessary functions
from tts.tts import process_audio
from llm.main import generate_podcast_script
from llm.rag import podcast_from_rag
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
import os
from utils.utils import extract_text, calculate_bleu, calculate_rouge

## matplotlib and io are used to generate plots
import matplotlib.pyplot as plt
import io

# Define accents for HOST and GUEST
host_accent = "com.au"  # Australian English for HOST
guest_accent = "co.uk"  # British English for GUEST

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


def evaluate_models():
    print("Evaluating models...")
    df = pd.read_csv("./podcast_samples/podcast_db.csv")

    # Initialize models
    initialize_models()

    # Initialize lists to store metrics
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

    random_podcast = df.sample(1)

    pdf_file = random_podcast["file"].values[0]
    podcast_file = random_podcast["podcast"].values[0]

    print(f"PDF file: {pdf_file}")
    print(f"Podcast file: {podcast_file}")

    with open(podcast_file, "r") as f:
        podcast_script = f.read()

    print(f"Starting evaluation...")
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

    # print(f"------ Base LLM ------")
    # print(
    #     f"Average inference time: {sum(inference_times['Base LLM']) / len(inference_times['Base LLM'])}"
    # )
    # print(
    #     f"Average ROUGE-1: {sum([r['rouge1'] for r in rouge_scores_each['Base LLM']]) / len(rouge_scores_each['Base LLM'])}"
    # )
    # print(
    #     f"Average ROUGE-2: {sum([r['rouge2'] for r in rouge_scores_each['Base LLM']]) / len(rouge_scores_each['Base LLM'])}"
    # )
    # print(
    #     f"Average ROUGE-L: {sum([r['rougeL'] for r in rouge_scores_each['Base LLM']]) / len(rouge_scores_each['Base LLM'])}"
    # )
    # print(
    #     f"Average BLEU: {sum(bleu_scores_each['Base LLM']) / len(bleu_scores_each['Base LLM'])}"
    # )

    # print(f"------ RAG ------")
    # print(
    #     f"Average inference time: {sum(inference_times['RAG']) / len(inference_times['RAG'])}"
    # )
    # print(
    #     f"Average ROUGE-1: {sum([r['rouge1'] for r in rouge_scores_each['RAG']]) / len(rouge_scores_each['RAG'])}"
    # )
    # print(
    #     f"Average ROUGE-2: {sum([r['rouge2'] for r in rouge_scores_each['RAG']]) / len(rouge_scores_each['RAG'])}"
    # )
    # print(
    #     f"Average ROUGE-L: {sum([r['rougeL'] for r in rouge_scores_each['RAG']]) / len(rouge_scores_each['RAG'])}"
    # )
    # print(
    #     f"Average BLEU: {sum(bleu_scores_each['RAG']) / len(bleu_scores_each['RAG'])}"
    # )

    # print(f"------ Custom LLM ------")
    # print(
    #     f"Average inference time: {sum(inference_times['Custom LLM']) / len(inference_times['Custom LLM'])}"
    # )
    # print(
    #     f"Average ROUGE-1: {sum([r['rouge1'] for r in rouge_scores_each['Custom LLM']]) / len(rouge_scores_each['Custom LLM'])}"
    # )
    # print(
    #     f"Average ROUGE-2: {sum([r['rouge2'] for r in rouge_scores_each['Custom LLM']]) / len(rouge_scores_each['Custom LLM'])}"
    # )
    # print(
    #     f"Average ROUGE-L: {sum([r['rougeL'] for r in rouge_scores_each['Custom LLM']]) / len(rouge_scores_each['Custom LLM'])}"
    # )
    # print(
    #     f"Average BLEU: {sum(bleu_scores_each['Custom LLM']) / len(bleu_scores_each['Custom LLM'])}"
    # )
    return {
        "inference_times": inference_times,
        "rouge_scores_each": rouge_scores_each,
        "bleu_scores_each": bleu_scores_each,
    }


import os


def evaluate_models_and_plot():
    # Evaluate models and get metrics
    results = evaluate_models()

    # Unpack metrics
    models = ["Base LLM", "RAG", "Custom LLM"]
    inference_times = [
        sum(results["inference_times"]["Base LLM"])
        / len(results["inference_times"]["Base LLM"]),
        sum(results["inference_times"]["RAG"]) / len(results["inference_times"]["RAG"]),
        sum(results["inference_times"]["Custom LLM"])
        / len(results["inference_times"]["Custom LLM"]),
    ]
    rouge1_scores = [
        sum([r["rouge1"] for r in results["rouge_scores_each"]["Base LLM"]])
        / len(results["rouge_scores_each"]["Base LLM"]),
        sum([r["rouge1"] for r in results["rouge_scores_each"]["RAG"]])
        / len(results["rouge_scores_each"]["RAG"]),
        sum([r["rouge1"] for r in results["rouge_scores_each"]["Custom LLM"]])
        / len(results["rouge_scores_each"]["Custom LLM"]),
    ]
    bleu_scores = [
        sum(results["bleu_scores_each"]["Base LLM"])
        / len(results["bleu_scores_each"]["Base LLM"]),
        sum(results["bleu_scores_each"]["RAG"])
        / len(results["bleu_scores_each"]["RAG"]),
        sum(results["bleu_scores_each"]["Custom LLM"])
        / len(results["bleu_scores_each"]["Custom LLM"]),
    ]

    # Create a table to display the results
    table = pd.DataFrame(
        {
            "Model": models,
            "Inference Time (s)": inference_times,
            "ROUGE-1 Score": rouge1_scores,
            "BLEU Score": bleu_scores,
        }
    )

    # Directory to save plots
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_files = []

    # Inference Time Plot
    fig, ax = plt.subplots()
    ax.bar(models, inference_times, color=["blue", "green", "orange"])
    ax.set_title("Inference Time (Seconds)")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Model")
    plot_path = os.path.join(plots_dir, "inference_time.png")
    plt.savefig(plot_path)
    plot_files.append(plot_path)

    # ROUGE-1 Scores Plot
    fig, ax = plt.subplots()
    ax.bar(models, rouge1_scores, color=["blue", "green", "orange"])
    ax.set_title("ROUGE-1 Scores")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plot_path = os.path.join(plots_dir, "rouge1_scores.png")
    plt.savefig(plot_path)
    plot_files.append(plot_path)

    # BLEU Scores Plot
    fig, ax = plt.subplots()
    ax.bar(models, bleu_scores, color=["blue", "green", "orange"])
    ax.set_title("BLEU Scores")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plot_path = os.path.join(plots_dir, "bleu_scores.png")
    plt.savefig(plot_path)
    plot_files.append(plot_path)

    return plot_files, table


def generate_podcast_with_feedback(pdf_file, model_type):
    # Ensure models are initialized
    initialize_models()

    yield "🔄 Generating podcast script...", None, None

    if model_type == "Base LLM":
        print("Using Base LLM model")
        script = generate_podcast_script(pdf_file, model=ollama_model)
    elif model_type == "RAG":
        print("Using RAG model")
        script = podcast_from_rag(
            pdf_file, model=ollama_model, embedding_model=embedding_model
        )
    elif model_type == "Custom LLM":
        print("Using Custom LLM model")
        article = extract_text(pdf_file)
        url = "http://localhost:5000/generate-script"
        # url = "http://host.docker.internal:5000/generate-script" --> For Docker container to host

        data = {"text": article}

        response = requests.post(url, json=data)
        response_data = response.json()

        script = response_data["script"]

    else:
        yield "❌ Error: Unknown model type selected.", None, None
        return

    yield "✅ Podcast script generated!", script, gr.update(visible=False)

    yield "🔄 Processing and combining audio...", None, gr.update(visible=False)
    output_audio_path = process_audio(script, host_accent, guest_accent)

    if os.path.exists(output_audio_path):
        yield "✅ Podcast audio generated successfully!", script, gr.update(
            value=output_audio_path, visible=True
        )
    else:
        yield "❌ Error: Audio file not generated.", script, gr.update(visible=False)


with gr.Blocks(css=".main-block {max-width: 800px; margin: auto;}") as interface:
    with gr.Tabs():
        with gr.Tab(label="Podcast Generator"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        <div style="text-align: center;">
                            <h1>🎙️ Podcast Generator</h1>
                            <p>Convert research papers into engaging podcasts!</p>
                        </div>
                        """
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(
                        label="📄 Upload Your PDF File", type="filepath"
                    )
                    model_selector = gr.Dropdown(
                        label="Select Model",
                        choices=["Base LLM", "RAG", "Custom LLM"],
                        value="Base LLM",
                    )
                    generate_button = gr.Button("🚀 Generate Podcast")
                    progress_bar = gr.Label(label="🛠️ Progress Updates")
                with gr.Column(scale=2):
                    progress_text = gr.Textbox(
                        label="📜 Script",
                        placeholder="The generated podcast script will appear here...",
                        lines=10,
                    )
                    audio_player = gr.Audio(
                        label="🔊 Generated Podcast", visible=False, interactive=False
                    )

            generate_button.click(
                generate_podcast_with_feedback,
                inputs=[pdf_input, model_selector],
                outputs=[progress_bar, progress_text, audio_player],
            )

        with gr.Tab(label="Model Comparison"):
            with gr.Row():
                gr.Markdown(
                    """
                    <h2>📊 Model Comparative Analysis</h2>
                    <p>Visualize performance metrics of different models with plots and a summary table.</p>
                    """
                )
            with gr.Row():
                evaluate_button = gr.Button("📈 Evaluate and Plot")
            with gr.Row():
                inference_plot = gr.Image(label="Inference Times")
                rouge1_plot = gr.Image(label="ROUGE-1 Scores")
                bleu_plot = gr.Image(label="BLEU Scores")
            with gr.Row():
                results_table = gr.Dataframe(
                    headers=[
                        "Model",
                        "Inference Time (s)",
                        "ROUGE-1 Score",
                        "BLEU Score",
                    ],
                    interactive=False,
                    label="📋 Evaluation Metrics Table",
                )

            def load_plots_and_table():
                plot_files, table = evaluate_models_and_plot()
                return plot_files[0], plot_files[1], plot_files[2], table

            # Link evaluation and plotting to button
            evaluate_button.click(
                load_plots_and_table,
                inputs=[],
                outputs=[inference_plot, rouge1_plot, bleu_plot, results_table],
            )


# Initialize models when the app starts
initialize_models()

interface.launch(debug=True)
