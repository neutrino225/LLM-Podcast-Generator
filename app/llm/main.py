import re
from langchain_core.prompts import ChatPromptTemplate
from utils.utils import extract_text
import os


def strip_markdown(text):
    # Remove Markdown headings, bold, italics, etc.
    clean_text = re.sub(r"([*_#>`\-\+])", "", text)
    return clean_text.strip()


def generate_podcast_script(file_path, model):
    template = """
    You are an expert podcast generator. You have been asked to create a podcast episode on the given pdf. The episode should be informative and engaging.

    Here is the content of the pdf:
    {content}

    Please create a podcast episode on this topic without any headings and clearly distinguish between host and guest parts use HOST and GUEST. The audience is interested in learning more about this topic.
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the LLM
    chain = prompt | model

    # Extract text from the PDF
    text = extract_text(file_path)

    # Generate the podcast script using the LLM
    response = chain.invoke(input={"content": text})

    # Post-process the response
    clean_response = strip_markdown(response)

    return clean_response


def generate_podcast_from_text(text, model):
    template = """
    You are an expert podcast generator. You have been asked to create a podcast episode on the given text. The episode should be informative and engaging.

    Here is the content of the text:
    {content}

    Please create a podcast episode on this topic without any headings and clearly distinguish between host and guest parts use HOST and GUEST. The audience is interested in learning more about this topic.
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the LLM
    chain = prompt | model

    # Generate the podcast script using the LLM
    response = chain.invoke(input={"content": text})

    # Post-process the response
    clean_response = strip_markdown(response)

    return clean_response
