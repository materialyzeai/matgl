# syntax=docker/dockerfile:1
#
# MatGL installed from this source tree. linux/amd64 and linux/arm64.
#
# Build from the repository root. The build context has no .git, so the version
# must be passed explicitly:
#   docker build -f docker/Dockerfile \
#       --build-arg MATGL_VERSION="$(uvx --from setuptools-scm python -m setuptools_scm)" \
#       -t materialyzeai/matgl:latest .
#
# Published by .github/workflows/docker.yml on every release.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm

ARG MATGL_VERSION

RUN --mount=type=bind,source=.,target=/tmp/matgl,rw \
    test -n "${MATGL_VERSION}" || { echo "MATGL_VERSION build-arg is required" >&2; exit 1; }; \
    SETUPTOOLS_SCM_PRETEND_VERSION="${MATGL_VERSION}" uv pip install --system --no-cache /tmp/matgl
