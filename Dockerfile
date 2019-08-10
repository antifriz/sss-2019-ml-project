# see https://hub.docker.com/_/python
FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY dockerfiles .

CMD [ "python", "app.py" ]
