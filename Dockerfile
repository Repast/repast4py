FROM python:3.8

RUN apt-get update && \
    apt-get install -y  mpich \
        && rm -rf /var/lib/apt/lists/*

# Set the current directory to /app
WORKDIR /repast4py

# Install the python requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

# Set the PYTHONPATH to include the /repast4py folder which contains the core folder
ENV PYTHONPATH=/repast4py/src

COPY ./setup.py ./setup.py
COPY ./src ./src
COPY ./tests ./tests

RUN CC=mpicc CXX=mpicxx python setup.py build_ext --inplace

CMD ["mpirun", "-n","8", "python", "src/zombies/zombies.py", "src/zombies/zombie_model.props"]
#CMD bash
