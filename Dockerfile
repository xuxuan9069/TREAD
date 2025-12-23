FROM python:3.8-slim

WORKDIR /TREAD
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python3", "train.py", "--config", "config.json"]
