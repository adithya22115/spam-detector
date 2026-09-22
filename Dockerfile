# Spam detection API — build with:  docker build -t spam-detector .
# Run with:                       docker run -p 5000:5000 spam-detector
FROM python:3.13-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the trained model
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

ENV FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    FLASK_DEBUG=0 \
    MODEL_PATH=models/spam_model.pkl

EXPOSE 5000

CMD ["python", "app/app.py"]