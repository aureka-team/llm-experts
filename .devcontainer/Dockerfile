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

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local" sh

WORKDIR /tmp

COPY .devcontainer/requirements.txt .
RUN --mount=type=cache,target=/root/.cache uv pip install -r requirements.txt --system

COPY ./requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache uv pip install -r requirements.txt --system

COPY ./llm_experts /tmp/llm_experts
COPY ./pyproject.toml /tmp/pyproject.toml
RUN --mount=type=cache,target=/root/.cache \
    --mount=source=.git,target=.git,type=bind \
    uv pip install . --no-deps --system

RUN rm -rf /tmp/*

ARG DEVCONTAINER_USER
RUN usermod -aG sudo $DEVCONTAINER_USER \
    && passwd -d $DEVCONTAINER_USER

ENV SHELL=/usr/bin/zsh
RUN chsh $DEVCONTAINER_USER -s $SHELL

WORKDIR /workspace

USER $DEVCONTAINER_USER
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
