FROM python:3.11-slim

WORKDIR /app

# 1) copy your FastAPI code
COPY reddit_app/ ./reddit_app/
# 2) copy your serialized pipeline
COPY reddit_model_pipeline.joblib ./
# 3) copy deps and server script
COPY requirements.txt   ./
COPY server.sh          ./

ENV GOOGLE_APPLICATION_CREDENTIALS="./secrets/credentials"

RUN pip install --upgrade pip \
 && pip install -r requirements.txt

EXPOSE 8080

RUN chmod +x server.sh
ENTRYPOINT ["./server.sh"]