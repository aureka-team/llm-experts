FROM ubuntu:noble

ARG PYTHON_VERSION
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    vim \
    zsh \
    wget \
    curl \
    sudo \
    htop \
    unzip \
    git-core \
    openssh-client \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-setuptools \
    ca-certificates \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/tmp/* /var/lib/apt/lists/*

RUN rm /usr/lib/python${PYTHON_VERSION}/EXTERNALLY-MANAGED
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

WORKDIR /tmp

ARG PIP_VERSION
RUN wget --no-check-certificate \
    https://github.com/pypa/pip/archive/refs/tags/${PIP_VERSION}.zip \
    && unzip ${PIP_VERSION}.zip \
    && cd pip-${PIP_VERSION} \
    && python setup.py install

COPY .devcontainer/requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

COPY ./requirements.txt requirements.txt
RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache \
    mkdir -p -m 0600 ~/.ssh \
    && ssh-keyscan github.com >> ~/.ssh/known_hosts \
    && pip install -r requirements.txt

COPY ./llm_experts /tmp/llm_experts
COPY ./pyproject.toml /tmp/pyproject.toml
RUN --mount=type=cache,target=/root/.cache \
    --mount=source=.git,target=.git,type=bind \
    pip install . --no-deps

RUN rm -rf /tmp/*

ARG DEVCONTAINER_USER
RUN usermod -aG sudo $DEVCONTAINER_USER \
    && passwd -d $DEVCONTAINER_USER

ENV SHELL=/usr/bin/zsh
RUN chsh $DEVCONTAINER_USER -s $SHELL

WORKDIR /workspace

USER $DEVCONTAINER_USER
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
