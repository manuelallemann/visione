FROM python:3.10

COPY . /usr/src/app
WORKDIR /usr/src/app

RUN pip install --no-cache-dir h5py==3.10.0 more_itertools==10.2.0 numpy==1.23.5 pandas==1.5.3 pyyaml==6.0.1 scikit-image==0.19.3 tqdm==4.48.2

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/"