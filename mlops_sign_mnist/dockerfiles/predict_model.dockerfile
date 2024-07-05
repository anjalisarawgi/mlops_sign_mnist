# Base image
FROM python:3.9-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY mlops_sign_mnist/requirements.txt mlops_sign_mnist/requirements.txt
COPY mlops_sign_mnist/pyproject.toml mlops_sign_mnist/pyproject.toml
COPY mlops_sign_mnist/ mlops_sign_mnist/
COPY mlops_sign_mnist/data/ mlops_sign_mnist/data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "mlops_sign_mnist/predict_model.py"]