#!/bin/bash
source /root/.bashrc 2>/dev/null || true
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
git config --global --add safe.directory /workspace/CrazyPlayGround 2>/dev/null || true
exec "$@"
