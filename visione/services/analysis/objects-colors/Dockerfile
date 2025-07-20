FROM python:3.10

COPY . /usr/src/app
WORKDIR /usr/src/app

RUN pip install --no-cache-dir h5py==3.10.0 more_itertools==10.2.0 numpy==1.23.5 pandas==1.5.3 pyyaml==6.0.1 scikit-image==0.19.3 tqdm
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/"