FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

ADD . /www
WORKDIR /www

RUN apt-get update 
RUN apt-get install -y libsndfile1
RUN apt-get install -y git
# RUN apt-get update 
# RUN apt-get install -y gcc
# RUN apt-get install -y build-essential python3.8-dev

RUN pip install pyuwsgi
RUN pip install -r requirements.txt

RUN cd /tmp;git clone https://github.com/huggingface/transformers.git
RUN cd /tmp/transformers; pip install .


CMD uwsgi uwsgi.ini