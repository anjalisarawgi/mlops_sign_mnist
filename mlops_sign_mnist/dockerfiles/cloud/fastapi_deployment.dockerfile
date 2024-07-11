# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app/mlops_sign_mnist

# Copy the current directory contents into the container at /app
COPY mlops_sign_mnist/main.py /app/mlops_sign_mnist/main.py
COPY requirements_deployment.txt /app/requirements_deployment.txt
COPY models/sign_language_mnist_model.pth /app/models/sign_language_mnist_model.pth
COPY mlops_sign_mnist/models/model.py /app/mlops_sign_mnist/models/model.py

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --timeout=300 -r ../requirements_deployment.txt


EXPOSE 8080

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]