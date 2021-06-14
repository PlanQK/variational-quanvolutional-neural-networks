FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

WORKDIR /experiments
RUN apt-get update && apt-get install -y \
    build-essential supervisor git cmake

ADD experiments /experiments
RUN pip install -r requirements.txt

ADD tensorboard.conf /etc/supervisor/conf.d/tensorboard.conf

CMD supervisord -n
