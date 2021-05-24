# Only for myself
FROM python:3.8-slim

WORKDIR /root/xai/ 

COPY requirements.txt /root/xai/requirements.txt

RUN apt-get update && \
    apt-get install -y git &&\
    apt-get upgrade -y  



