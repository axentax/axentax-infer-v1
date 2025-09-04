FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

RUN pip install --no-cache-dir --prefer-binary \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio \
 && pip install --no-cache-dir \
    transformers peft datasets accelerate

# 必要パッケージ
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      curl ca-certificates git bash coreutils build-essential \
  && rm -rf /var/lib/apt/lists/*

# nvm をシステム共通ディレクトリに配置
ENV NVM_DIR=/usr/local/nvm
RUN mkdir -p "$NVM_DIR" \
 && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Node.js v22.12.0 をインストール & デフォルト化
ENV NODE_VERSION=22.12.0
RUN . "$NVM_DIR/nvm.sh" \
 && nvm install $NODE_VERSION \
 && nvm alias default $NODE_VERSION \
 && nvm use default

# すべてのユーザーで node/npm/npx を使えるよう PATH を固定
ENV PATH="$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH"

# 一般ユーザーからも読めるように権限を付与（任意：読み取りで十分）
RUN chmod -R a+rX "$NVM_DIR"

RUN npm install -g @anthropic-ai/claude-code

RUN pip install --upgrade huggingface_hub

# 非対話設定
ENV DEBIAN_FRONTEND=noninteractive

# 1) sudo と必要最低限のツール
RUN apt-get update && apt-get install -y sudo ca-certificates bash \
 && rm -rf /var/lib/apt/lists/*

# 2) 一般ユーザー ift を作成して sudo グループへ
#    ※UID/GID をホストに合わせたい場合は --uid/--gid を明示
RUN adduser --disabled-password --gecos "" ift \
 && usermod -aG sudo ift

# 3) パスワードなし sudo
RUN echo 'ift ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/010-ift-nopasswd \
 && chmod 0440 /etc/sudoers.d/010-ift-nopasswd \
 && visudo -cf /etc/sudoers.d/010-ift-nopasswd

# RUN groupadd -g 1000 hostgroup && \
#     useradd -m -u 1000 -g 1000 ift
RUN echo "export PS1='\\u@ft:\\w\\$ '" >> /home/ift/.bashrc \
 && echo "ft" > /etc/hostname

WORKDIR /work
