FROM ubuntu:20.04

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/Madrid"

RUN apt-get update && \
	apt-get install -y \
        build-essential \
        git \
        autoconf \
        libtool \
        pkg-config \
        g++ \
        gcc-9 \
        wget \
        gfortran

# Install updated version of cmake
ARG CMAKE_VERSION=3.23.2
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /usr/bin/cmake \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
      && rm /tmp/cmake-install.sh
ENV PATH="/usr/bin/cmake/bin:${PATH}"

COPY . /app
WORKDIR /app

RUN chmod +x ./build.sh && ./build.sh -b Release
ENTRYPOINT [ "./mercator" ]