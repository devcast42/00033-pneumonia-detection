FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY project/requirements.txt ./project/requirements.txt
RUN python -m pip install --no-cache-dir -r project/requirements.txt

COPY . .

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "project.api.main:app", "--host", "0.0.0.0", "--port", "8002"]
