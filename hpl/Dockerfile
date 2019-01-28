# code from https://www.pugetsystems.com/labs/hpc/How-to-Run-an-Optimized-HPL-Linpack-Benchmark-on-AMD-Ryzen-Threadripper----2990WX-32-core-Performance-1291/
# and https://www.howtoforge.com/tutorial/hpl-high-performance-linpack-benchmark-raspberry-pi/

FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install build-essential emacs -y
RUN apt-get install hwloc libhwloc-dev libevent-dev -y
RUN apt-get install autoconf automake gfortran -y

RUN apt install libopenmpi-dev -y
RUN apt install openssh-server -y

RUN apt-get install libatlas-base-dev gfortran -y

COPY hpl-2.3.tar.gz /hpl-2.3.tar.gz
RUN tar xf hpl-2.3.tar.gz
RUN mv hpl-2.3 hpl

# COPY hpl-2.2.tar.gz /hpl-2.2.tar.gz
# RUN tar xf hpl-2.2.tar.gz
# RUN mv hpl-2.2 hpl

# wget http://www.netlib.org/benchmark/hpl/hpl-2.2.tar.gz

COPY Make.ubuntu /hpl/Make.ubuntu
WORKDIR "/hpl"
RUN make arch=ubuntu
WORKDIR "/hpl/bin/ubuntu"

COPY HPL.dat HPL.dat

CMD ./xhpl
