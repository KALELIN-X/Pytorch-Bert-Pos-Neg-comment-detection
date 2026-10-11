# Optional API container. Build context is the repository root.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt
# Copy only serving source; local .env, private CSVs and artifacts stay outside image.
COPY src ./src
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
