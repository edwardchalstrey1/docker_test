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

# CMD mkdir ~/AMD-HPL

# The below compiles but doesnt run
# COPY hello.c /hello.c
# CMD mpicc hello.c -o hello
# CMD mpirun --allow-run-as-root -np 4 hello

COPY hpl-2.2.tar.gz /hpl-2.2.tar.gz
RUN tar xf hpl-2.2.tar.gz
RUN mv hpl-2.2 hpl

COPY Make.ubuntu /hpl/Make.ubuntu

COPY commands.sh /commands.sh
COPY commands2.sh /commands2.sh
COPY HPL.dat HPL.dat

CMD bash commands.sh
# CMD bash commands2.sh
