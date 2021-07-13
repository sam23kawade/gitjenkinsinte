FROM ubuntu:18.04

 

RUN apt-get update -y
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
EXPOSE 5000

 

COPY ./requirements.txt /app/requirements.txt

 

RUN pip3 install -r /app/requirements.txt
WORKDIR /app

 

COPY . /app

 

ENTRYPOINT [ "python3" ]

 

CMD [ "test2.py" ]
