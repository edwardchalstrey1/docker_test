FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN pip3 install numpy

COPY benchmarks.py /benchmarks.py

CMD python3 benchmarks.py
