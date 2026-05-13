FROM nvcr.io/nvidia/isaac-lab:2.3.0
SHELL ["/bin/bash", "-lc"]

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV ISAACLAB_PATH=/workspace/isaaclab
ENV ISAACSIM_ROOT_PATH=/isaac-sim
ENV PYTHONPATH="/workspace/CrazyPlayGround:${PYTHONPATH}"
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

USER root
WORKDIR /workspace/CrazyPlayGround

# Copy source first for layer caching
COPY source/CrazyPlayGround /workspace/CrazyPlayGround/source/CrazyPlayGround

# Install wandb + CrazyPlayGround at build time (no runtime pip installs)
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir wandb==0.16.6 && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
        -e /workspace/CrazyPlayGround/source/CrazyPlayGround

# Copy the rest of the project
COPY . /workspace/CrazyPlayGround

# Entrypoint
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> /root/.bashrc && \
    echo "export ISAACSIM_ROOT_PATH=${ISAACSIM_ROOT_PATH}" >> /root/.bashrc && \
    echo "export PYTHONPATH=/workspace/CrazyPlayGround:\${PYTHONPATH}" >> /root/.bashrc && \
    echo "alias isaaclab=\${ISAACLAB_PATH}/isaaclab.sh" >> /root/.bashrc

ENTRYPOINT ["/usr/local/bin/entrypoint.sh", "bash"]
