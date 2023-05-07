FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=100

RUN apt-get update && apt-get upgrade -y
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F39F6D246A59DDE01C866C425A78330DFF0A946E
RUN apt-get update && \
    apt-get upgrade && \
    apt-get -y install locales git vim curl build-essential libffi-dev libssl-dev zlib1g-dev liblzma-dev libbz2-dev libreadline-dev libsqlite3-dev libopencv-dev tk-dev git pip

WORKDIR /root/workdir/

# pyenv
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
RUN  bash -c 'source ~/.bashrc;'

RUN /root/.pyenv/bin/pyenv install 3.10.6 && \
    /root/.pyenv/bin/pyenv local 3.10.6

RUN /root/.pyenv/shims/pip install --upgrade pip && \
    /root/.pyenv/shims/pip install poetry

COPY pyproject.toml ./

COPY .torch_wheels ./.torch_wheels

RUN ls /root/.pyenv/shims

RUN  /root/.pyenv/shims/poetry config virtualenvs.create false && \
    /root/.pyenv/shims/poetry install


CMD ["/bin/bash"]
