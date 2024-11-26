# Use a base image with Miniconda pre-installed
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the repository into the container
COPY . .

# Create the environment using the env.yml file
RUN conda env create -f env.yml

# Activate the environment
RUN echo "conda activate dlq_py_env" > ~/.bashrc
ENV PATH=/opt/conda/envs/dlq_py_env/bin:$PATH

# Run tests when the container starts
CMD ["pytest"]
