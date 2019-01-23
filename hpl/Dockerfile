# code from https://www.pugetsystems.com/labs/hpc/How-to-Run-an-Optimized-HPL-Linpack-Benchmark-on-AMD-Ryzen-Threadripper----2990WX-32-core-Performance-1291/

FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install build-essential emacs -y
RUN apt-get install hwloc libhwloc-dev libevent-dev -y
RUN apt-get install autoconf automake gfortran -y

# RUN dpkg --configure -a
# RUN apt install lam4-dev -f -y
# RUN apt install limbpich-dev -y
RUN apt install libopenmpi-dev -y
RUN apt install openssh-server -y

CMD mkdir ~/AMD-HPL
CMD mkdir ~/projects
CMD mkdir ~/projects/hpl-build

COPY openmpi-3.1.3.tar.gz /projects/hpl-build/openmpi-3.1.3.tar.gz
COPY mpi-env.sh /AMD-HPL/mpi-env.sh

CMD cd projects/hpl-build

CMD tar xf openmpi-3.1.3.tar.gz
CMD cd openmpi-3.1.3/

CMD CFLAGS="-Ofast -march=native" ./configure --prefix=$HOME/AMD-HPL
#CMD make -j 32
CMD make
CMD make install

CMD export MPI_HOME=$HOME/AMD-HPL/openmpi-3.1
CMD export PATH=$PATH:$MPI_HOME/bin
CMD export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_HOME/lib

RUN /bin/bash -c "source AMD-HPL/mpi-env.sh"

COPY hello.c /hello.c

CMD cd ../../..

# CMD ls
# RUN mpicc hello.c -o hello
# RUN mpirun -np 4 hello

CMD mpicc hello.c -o hello
CMD mpirun --allow-run-as-root -np 4 hello
