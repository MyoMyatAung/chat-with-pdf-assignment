FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt  # Create requirements.txt with pip freeze > requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]