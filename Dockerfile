# syntax=docker/dockerfile:1

FROM python:3.7

WORKDIR /app

EXPOSE 5001

COPY requirements.txt requirements.txt

COPY main.py main.py
COPY my_model_roi_nasnet_trained.h5 my_model_roi_nasnet_trained.h5


RUN pip install -r requirements.txt

#RUN mkdir -p /app/data


CMD python main.py
