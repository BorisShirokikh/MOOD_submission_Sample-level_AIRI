FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
	git wget screen tmux bzip2 gcc

# paths
RUN mkdir /workspace
RUN mkdir /workspace/resources

RUN mkdir /mnt/data
RUN mkdir /mnt/pred

ADD scripts /workspace/
ADD resources /workspace/resources/

RUN chmod +x /workspace/*.sh

# miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN chmod +x miniconda.sh
RUN ./miniconda.sh -b -p /miniconda3
RUN chmod -R 777 /miniconda3
RUN rm ./miniconda.sh
ENV PATH="/miniconda3/bin:${PATH}"
RUN conda install -y python=3.10

RUN pip install joblib
RUN pip install numpy
RUN pip install nibabel
RUN pip install imops
# imops silently requires skimage:(((
RUN pip install scikit-image
RUN pip install scikit-learn
RUN pip install tqdm

ENTRYPOINT ["/bin/bash"]
