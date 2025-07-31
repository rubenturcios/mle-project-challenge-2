
FROM python:3.12


WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./model /code/model
COPY ./data/zipcode_demographics.csv /code/data/zipcode_demographics.csv

#RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install -r /code/requirements.txt

COPY ./src /code/app
