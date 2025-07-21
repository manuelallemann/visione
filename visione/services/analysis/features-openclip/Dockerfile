FROM docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/"
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface
RUN mkdir -p /cache/huggingface && chmod -R 777 /cache/huggingface

COPY . .
CMD ["python", "-u", "service.py"]