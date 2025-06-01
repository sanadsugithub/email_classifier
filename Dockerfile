FROM python:3.11-slim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir sentencepiece

RUN pip install --no-cache-dir scikit-learn==1.6.1

ENV TRANSFORMERS_CACHE=/tmp/huggingface

COPY . /app
WORKDIR /app

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
