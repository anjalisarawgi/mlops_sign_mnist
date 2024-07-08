# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY mlops_sign_mnist/requirements.txt mlops_sign_mnist/requirements.txt
COPY mlops_sign_mnist/pyproject.toml mlops_sign_mnist/pyproject.toml
COPY mlops_sign_mnist/ mlops_sign_mnist/
COPY mlops_sign_mnist/data/ mlops_sign_mnist/data/
COPY mlops_sign_mnist/configs/ mlops_sign_mnist/configs/
COPY mlops_sign_mnist/models/ mlops_sign_mnist/models/
COPY mlops_sign_mnist/reports/figures mlops_sign_mnist/reports/figures/


WORKDIR /
RUN pip install -r mlops_sign_mnist/requirements.txt --no-cache-dir
RUN pip install ./mlops_sign_mnist --no-deps --no-cache-dir

ENV WANDB_API_KEY=1b7ecebed8a240adafb51f6be5c3365569eda1fb

# not sure 
ENV WANDB_MODE=online
ENV HYDRA_FULL_ERROR=1

ENTRYPOINT ["python", "-u", "mlops_sign_mnist/mlops_sign_mnist/predict_model.py"]