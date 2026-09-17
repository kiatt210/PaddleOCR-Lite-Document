FROM python:3.11-slim-bookworm

## Add dependencies
RUN apt-get update && apt install --yes cmake wget build-essential gcc g++ libprotobuf-dev protobuf-compiler libopencv-dev git libgrpc++-dev protobuf-compiler-grpc
RUN pip install opencv-contrib-python
## Install ncnn
RUN git clone --depth=1 https://github.com/Tencent/ncnn.git && \
    cd ncnn && \
    mkdir build && \
    cd build && \
    cmake -D NCNN_DISABLE_RTTI=OFF -D NCNN_BUILD_TOOLS=ON \
    -D CMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake .. && \
    make -j4 && \
    make install && \
    mkdir /usr/local/lib/ncnn && \
    cp -r install/include/ncnn /usr/local/include/ncnn && \
    cp -r install/lib/libncnn.a /usr/local/lib/ncnn/libncnn.a

WORKDIR /app
