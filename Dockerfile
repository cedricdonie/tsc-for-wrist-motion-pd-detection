# Docker container for GPU training/inference with TensorFlow and sktime-dl
# Author(s): Cedric Donie (cedricdonie@gmail.com)

FROM tensorflow/tensorflow:2.5.0-gpu
RUN apt-get update
RUN apt-get install -y git
RUN python -m pip install --upgrade pip
RUN pip install Cython==0.29.14

# Don't mess with the TensorFlow version already in the base image
COPY requirements.txt requirements.txt
RUN grep -v "tensorflow|tensorboard" requirements.txt > requirements_no_tensorflow.txt
RUN pip install -r requirements_no_tensorflow.txt
COPY requirements-no-deps.txt requirements-no-deps.txt
RUN pip install -r requirements-no-deps.txt --no-deps --ignore-requires-python

RUN apt-get install -y parallel

# Install entire source as package (no need to upload code separately to cloud)
COPY setup.py work/
COPY src work/src
COPY test work/test
RUN pip install -e work
WORKDIR /work

ARG GIT_COMMIT=unspecified
LABEL org.opencontainers.image.revision=$GIT_COMMIT