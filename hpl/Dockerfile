FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install build-essential emacs -y
RUN apt install libopenmpi-dev -y
RUN apt install openssh-server -y
RUN apt-get install libatlas-base-dev gfortran -y

COPY hpl-2.3.tar.gz /hpl-2.3.tar.gz
RUN tar xf hpl-2.3.tar.gz
RUN mv hpl-2.3 hpl

COPY Make.ubuntu /hpl/Make.ubuntu
WORKDIR "/hpl"
RUN make arch=ubuntu
WORKDIR "/hpl/bin/ubuntu"

COPY HPL.dat HPL.dat

CMD mpirun -np 4 --allow-run-as-root ./xhpl
