# code from https://www.pugetsystems.com/labs/hpc/How-to-Run-an-Optimized-HPL-Linpack-Benchmark-on-AMD-Ryzen-Threadripper----2990WX-32-core-Performance-1291/

FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install build-essential emacs -y
RUN apt-get install hwloc libhwloc-dev libevent-dev -y
RUN apt-get install autoconf automake gfortran -y

CMD mkdir ~/AMD-HPL
CMD mkdir ~/projects/hpl-build

COPY openmpi-3.1.3.tar.gz /projects/hpl-build/openmpi-3.1.3.tar.gz
COPY mpi-env.sh /AMD-HPL/mpi-env.sh

CMD cd projects/hpl-build

CMD tar xf openmpi-3.1.3.tar.gz
CMD cd openmpi-3.1.3/

CMD CFLAGS="-Ofast -march=native" ./configure --prefix=$HOME/AMD-HPL
CMD make -j 32
CMD make install

CMD export MPI_HOME=$HOME/AMD-HPL/openmpi-3.1
CMD export PATH=$PATH:$MPI_HOME/bin
CMD export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_HOME/lib

RUN /bin/bash -c "source AMD-HPL/mpi-env.sh"

COPY simple-mpi-test.c /simple-mpi-test.c

# CMD ls
RUN /bin/bash -c "mpicc -o simple-mpi-test simple-mpi-test."
RUN /bin/bash -c "mpirun -np 4 simple-mpi-test"
