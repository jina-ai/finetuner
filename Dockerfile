FROM registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda11.2-cudnn8

RUN pip install -i https://mirrors.cloud.tencent.com/pypi/simple jina

WORKDIR /workspace

ENTRYPOINT ["/bin/bash"]
