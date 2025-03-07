FROM python:3.12.4

COPY requirements.txt /app/requirements.txt

RUN apt-get update && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . ./vecragbot

WORKDIR /vecragbot

CMD ["python3", "app.py"]