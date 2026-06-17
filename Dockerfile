FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY CMakeLists.txt pyproject.toml ./
COPY include/ include/
COPY src/ src/
COPY python/ python/
COPY streamlit_app.py run.py app.py ./

RUN pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
