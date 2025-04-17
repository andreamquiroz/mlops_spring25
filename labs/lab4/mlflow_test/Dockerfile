# Use an official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install MLflow
RUN pip install mlflow

# Expose port 5000 for the MLflow server
EXPOSE 5000

# Launch the MLflow server when the container starts
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]