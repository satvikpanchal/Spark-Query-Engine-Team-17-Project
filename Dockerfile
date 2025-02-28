# --------------------------------------------------------------
# Dockerfile: Spark, PostgreSQL, Python3, PyTorch (CPU)
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
        postgresql postgresql-contrib \
        libssl-dev \
        && apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # Set python3 as the default python
    RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
        update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
    
    # Install Python libraries
    # - CPU-only PyTorch (version is an example; you can adjust to current stable release)
    # - PySpark for Spark integration
    # - Additional DS/ML libraries as needed
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
    
    # Expose PostgreSQL and Spark ports
    EXPOSE 5432 7077 8080 4040
    
    WORKDIR /workdir
    
    # Optionally copy in your project code, if you have it locally:
    # COPY . /workdir
    
    # Set the default command to a bash shell
    CMD ["/bin/bash"]
    