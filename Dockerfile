FROM ubuntu:20.04

# set up python
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    python3 -m pip --no-cache-dir install --upgrade pip setuptools && \
    ln -s $(which python3) /usr/local/bin/python

# install object detection api dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y protobuf-compiler python-pil python-lxml python-tk && \
    python3 -m pip install --no-cache-dir tensorflow \
    Cython \
    contextlib2 \
    pillow \
    lxml \
    jupyter \
    matplotlib \
    tf_slim

# download and set up TensorFlow models repozitory
RUN apt-get install -y git && \
    git clone https://github.com/tensorflow/models.git && \
    cd models/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    pip install . && \
    cd / && \
    rm -rf models

# set default command to bash
CMD ["/bin/bash"]