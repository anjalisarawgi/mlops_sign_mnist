# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_sign_mnist/ mlops_sign_mnist/
COPY data/ data/
COPY configs/ configs/
COPY models/ models/
COPY reports/figures/ reports/figures/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENV WANDB_API_KEY=1b7ecebed8a240adafb51f6be5c3365569eda1fb

ENTRYPOINT ["python", "-u", "mlops_sign_mnist/train_model.py"]
