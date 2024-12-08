# Use Miniconda3 as the base image
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Use environment.yml to create the conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "new_genai", "/bin/bash", "-c"]

# Expose gradio port
EXPOSE 7860

# Run the app
CMD ["conda", "run", "-n", "new_genai", "python", "app.py"]
