FROM python:3.12.4

COPY requirements.txt /vecragbot/requirements.txt

RUN apt-get update && \
    pip install --no-cache-dir -r /vecragbot/requirements.txt

COPY . ./vecragbot

WORKDIR /vecragbot

CMD ["python3", "app.py"]