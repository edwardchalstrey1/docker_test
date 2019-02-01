FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install build-essential emacs -y
RUN apt install libopenmpi-dev -y
RUN apt install openssh-server -y
RUN apt-get install libatlas-base-dev gfortran -y

COPY hpl-2.3.tar.gz /hpl-2.3.tar.gz
RUN tar xf hpl-2.3.tar.gz
RUN mv hpl-2.3 hpl

# Problem with doing it this way is that netlib was down when I tried this once
# wget http://www.netlib.org/benchmark/hpl/hpl-2.2.tar.gz

COPY Make.ubuntu /hpl/Make.ubuntu
WORKDIR "/hpl"
RUN make arch=ubuntu
WORKDIR "/hpl/bin/ubuntu"

COPY HPL.dat HPL.dat

CMD ./xhpl
