FROM python:3.10

COPY . /usr/src/app
WORKDIR /usr/src/app

RUN pip install --no-cache-dir -r requirements.txt
# Ensure torch and torchvision are installed (CPU version for compatibility)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/"