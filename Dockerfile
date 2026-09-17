FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal - pandas/numpy ship wheels for slim images already.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PORT=5000

EXPOSE 5000

# gunicorn, not the Flask dev server - see Procfile for the same command.
CMD ["sh", "-c", "gunicorn app:app --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT}"]
