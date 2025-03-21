# --------------------------------------------------------------
# Dockerfile: Spark, Python3, PyTorch (CPU-only)
# --------------------------------------------------------------
    FROM ubuntu:20.04

    ENV DEBIAN_FRONTEND=noninteractive
    
    # System packages
    RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl ca-certificates \
        openjdk-11-jdk-headless \
        python3-dev python3-pip \
        software-properties-common \
        gcc g++ make git \
        libssl-dev \
        && apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # Set python3 as the default python
    RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
        update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
    
    # Install Python libraries
    RUN pip install --no-cache-dir \
        torch==2.0.0 \
        torchvision==0.15.0 \
        torchaudio==2.0.0 \
        pyspark \
        numpy pandas scikit-learn jupyter
    
    # Environment variables for Spark
    ENV SPARK_VERSION=3.3.2
    ENV HADOOP_VERSION=3
    ENV SPARK_HOME=/opt/spark
    ENV PATH=$PATH:$SPARK_HOME/bin
    
    # Download and install Spark
    RUN wget --no-verbose \
        https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
        -O /tmp/spark.tgz && \
        mkdir -p /opt && \
        tar -xzf /tmp/spark.tgz -C /opt && \
        mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} $SPARK_HOME && \
        rm /tmp/spark.tgz
    
    # Expose Spark ports
    EXPOSE 7077 8080 4040
    
    WORKDIR /workdir
    
    # CMD
    CMD ["/bin/bash"]
    