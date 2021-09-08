FROM python:3.8

RUN apt-get update && \
    apt-get install -y  mpich \
        && rm -rf /var/lib/apt/lists/*

# Install the python requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

RUN echo "#!/bin/bash\n" > /startscript.sh
RUN echo "mkdir repos\n" >> /startscript.sh
RUN echo "cd repos\n" >> /startscript.sh
RUN echo "git clone https://github.com/networkx/networkx-metis.git\n" >> /startscript.sh
RUN echo "cd networkx-metis\n" >> /startscript.sh
RUN echo "python setup.py install\n" >> /startscript.sh
RUN echo "cd ../\n" >> /startscript.sh
RUN echo "rm -rf networkx-metis\n " >> /startscript.sh

RUN chmod +x /startscript.sh
CMD /startscript.sh

# Set the PYTHONPATH to include the /repast4py folder which contains the core folder
ENV PYTHONPATH=/repast4py/src