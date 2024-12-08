# LLM-Podcast-Generator

The **LLM-Podcast-Generator** project is an innovative solution that leverages advanced language models (LLMs) to transform research papers into engaging podcast scripts. By integrating various LLMs, including a base LLM, Retrieval-Augmented Generation (RAG) LLM, and a custom LLM, this project enables the seamless conversion of scientific research into audio content, providing an easier and more accessible way to consume complex research material.

## Features

- **Base LLM**: A foundational language model used for text processing and generating initial content.
- **RAG LLM**: Combines retrieval-based models with generative models to enhance content generation, ensuring accuracy and relevance.
- **Custom LLM**: Tailored for generating podcasts from research papers, trained specifically to transform academic text into clear and concise scripts suitable for audio narration.
- **API Integration**: Communicates with a locally running API for additional processing and script generation.
- **Text-to-Speech (TTS)**: Converts the generated podcast scripts into high-quality audio, ready for publishing.

## Requirements

To run this project, make sure you have the following installed:

- Python 3.x
- Conda (for environment management)
- Docker (for containerization, optional but recommended)

### Dependencies

This project uses both an `environment.yml` and `requirements.txt` file for managing dependencies. The `environment.yml` is used to set up the Conda environment, and `requirements.txt` includes Python packages that need to be installed.

## Setup

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/neutrino225/LLM-Podcast-Generator.git
cd LLM-Podcast-Generator
```

### 2. Conda Environment Setup
Create the Conda environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
```

Activate the Conda environment:
```bash
conda activate new_genai
```

### 4. Running the Application
Run the application using the following command:
```bash
cd app
python app.py
```


## API
The app also sends requests to a local API running on localhost:5000. Make sure the API server is running before starting the app.

### Example of API Request
The app will send a POST request to the localhost:5000/generate-script endpoint with the following JSON payload:

```json
{
  "script": "Article content extracted from the PDF file."
}
```

The response will contain the generated podcast script:
```json
{
  "script": "Generated podcast script based on the article."
}
```


## Contributing
If you would like to contribute to this project, feel free to open an issue or submit a pull request. Please make sure to follow the coding standards and provide a detailed explanation of the changes made.

## License
This project is licensed under the MIT License - see the LICENSE file for details.