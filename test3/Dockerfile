FROM python:3

RUN apt-get update
RUN apt install libopenmpi-dev -y
RUN pip3 install numpy
RUN pip3 install mpi4py

COPY benchmarks.py /benchmarks.py

CMD mpirun -n 4 --allow-run-as-root python3 benchmarks.py
