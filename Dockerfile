FROM continuumio/anaconda3

WORKDIR /root/xai/ 

COPY requirements.txt /root/xai/requirements.txt

RUN apt-get update && \
    apt-get install -y git &&\
    apt-get install -y build-essential && \
    apt-get install -y python-dev && \
    apt-get upgrade -y  

# Python

RUN apt-get install python3 python3-distutils python3-apt -y \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 \
  && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
  && python get-pip.py




