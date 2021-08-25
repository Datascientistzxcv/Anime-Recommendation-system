FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/

#WORKDIR usr/getty
COPY requirements.txt ./
RUN pip install  --use-feature=2020-resolver -r requirements.txt

VOLUME /home/getty
WORKDIR /home/getty