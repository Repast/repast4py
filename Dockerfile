FROM python:3.8

RUN apt-get update && \
    apt-get install -y  mpich \
        && rm -rf /var/lib/apt/lists/*

# Install the python requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

# RUN apt-get update \        
#     apt-get install -y git

RUN mkdir -p /repos && \
    cd /repos && \
    git clone --depth 1 https://github.com/networkx/networkx-metis.git && \
    cd /repos/networkx-metis && \
    python setup.py install

# Set the PYTHONPATH to include the /repast4py folder which contains the core folder
ENV PYTHONPATH=/repast4py/src