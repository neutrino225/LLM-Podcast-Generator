import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
import re


def strip_markdown(text):
    # Remove Markdown headings, bold, italics, etc.
    clean_text = re.sub(r"([*_#>`\-\+])", "", text)
    return clean_text.strip()


def podcast_from_rag(pdf_path, model, embedding_model):
    # Validate the path
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    # Step 1: Load the document
    new_doc = PyPDFLoader(pdf_path).load()

    # Step 2: Create the embedding model and vector store
    vector_store = Chroma.from_documents(documents=new_doc, embedding=embedding_model)

    # Step 3: Define the generic questions
    questions = [
        "What is the main research question?",
        "What are the key findings and their significance?",
        "What are the limitations of the study?",
        "How does this research contribute to the field?",
        "What are the implications for future research?",
        "What are the practical implications of this research?",
        "What are the key takeaways from this research?",
    ]

    # Step 4: Embed the questions
    question_embeddings = [
        embedding_model.embed_query(question) for question in questions
    ]

    # Step 5: Retrieve similar documents
    similar_docs = [
        vector_store.similarity_search_by_vector(embedding, k=5)
        for embedding in question_embeddings
    ]
    # Step 6: Create context for each question
    context = [" ".join([doc.page_content for doc in docs]) for docs in similar_docs]

    # Step 7: Define the refined prompt
    template = """
    You are an expert podcast generator. Your task is to create a podcast episode where a host interviews a guest. Generate responses for each question based on its context.

    These are the list of questions and their contexts:
    Questions and Contexts:
    {questions_and_contexts}

    Generate the conversation below and use the following format:
    HOST: ......
    GUEST: ......
    """

    # Format the input for batch processing
    questions_and_contexts = "\n".join(
        [
            f"Question: {question}\nContext: {ctx}"
            for question, ctx in zip(questions, context)
        ]
    )

    # Step 8: Create the model and process the input
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    print("Generating the podcast script...")
    response = chain.invoke(input={"questions_and_contexts": questions_and_contexts})

    # Post-process the response
    clean_response = strip_markdown(response)

    return clean_response


def generate_podcast_from_given_text(text, model, embedding_model):
    # Step 1: Create the embedding model and vector store
    vector_store = Chroma.from_texts(text, embedding=embedding_model)

    # Step 2: Define the generic questions
    questions = [
        "What is the main research question?",
        "What are the key findings and their significance?",
        "What are the limitations of the study?",
        "How does this research contribute to the field?",
        "What are the implications for future research?",
        "What are the practical implications of this research?",
        "What are the key takeaways from this research?",
    ]

    # Step 3: Embed the questions
    question_embeddings = [
        embedding_model.embed_query(question) for question in questions
    ]

    # Step 4: Retrieve similar documents
    similar_docs = [
        vector_store.similarity_search_by_vector(embedding, k=5)
        for embedding in question_embeddings
    ]
    # Step 5: Create context for each question
    context = [" ".join([doc.page_content for doc in docs]) for docs in similar_docs]

    # Step 6: Define the refined prompt
    template = """
    You are an expert podcast generator. Your task is to create a podcast episode where a host interviews a guest. Generate responses for each question based on its context.

    These are the list of questions and their contexts:
    Questions and Contexts:
    {questions_and_contexts}

    Generate the conversation below and use the following format:
    HOST: ......
    GUEST: ......
    """

    # Format the input for batch processing
    questions_and_contexts = "\n".join(
        [
            f"Question: {question}\nContext: {ctx}"
            for question, ctx in zip(questions, context)
        ]
    )

    # Step 7: Create the model and process the input
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    print("Generating the podcast script...")
    response = chain.invoke(input={"questions_and_contexts": questions_and_contexts})

    # Post-process the response
    clean_response = strip_markdown(response)

    return clean_response
