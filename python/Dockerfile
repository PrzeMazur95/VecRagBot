FROM python:3.12

RUN pip install qdrant-client sentence-transformers

WORKDIR /usr/app/src

COPY main.py ./
COPY food.csv ./
