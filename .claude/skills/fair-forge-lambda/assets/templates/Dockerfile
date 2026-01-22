# Fair-Forge Lambda Container
# Builds from local wheel (fair-forge is not available on PyPI)
#
# IMPORTANT: This Dockerfile is designed to be built with the repo root as context.
# The COPY paths for handler.py, run.py, and requirements.txt must be updated
# to use the full path from the repo root, e.g.:
#   COPY examples/metrics/aws-lambda/requirements.txt .
#   COPY examples/metrics/aws-lambda/handler.py examples/metrics/aws-lambda/run.py ${LAMBDA_TASK_ROOT}/

FROM public.ecr.aws/lambda/python:3.11 AS builder

# Install build tools for Python wheel
RUN pip install --upgrade pip build

# Copy the fair-forge source and build wheel
WORKDIR /build
COPY fair_forge/ ./fair_forge/
COPY pyproject.toml README.md LICENSE ./
RUN python -m build --wheel --outdir /dist

# Final Lambda image
FROM public.ecr.aws/lambda/python:3.11

# Copy built wheel from builder stage
COPY --from=builder /dist/*.whl /tmp/

# Upgrade pip
RUN pip install --upgrade pip

# Pre-install numpy and scipy with manylinux wheels (avoids compilation issues)
# The Lambda base image lacks compilers needed to build these from source.
# These versions have pre-built wheels compatible with the Lambda environment.
RUN pip install "numpy==1.26.4" "scipy==1.14.1" --target "${LAMBDA_TASK_ROOT}"

# Create constraints file to prevent scipy/numpy from being upgraded
# (fair-forge requires scipy>=1.14.1, but 1.14.1 is the last version with manylinux wheels
# that don't require compilation on Lambda's older glibc)
RUN echo "numpy==1.26.4" > /tmp/constraints.txt && \
    echo "scipy==1.14.1" >> /tmp/constraints.txt

# Install the wheel with the specified extras (MODULE_EXTRA build arg)
# Note: WHEEL=$(ls /tmp/*.whl) captures full path, so don't prefix with /tmp/
ARG MODULE_EXTRA=bestof
RUN WHEEL=$(ls /tmp/*.whl) && pip install "${WHEEL}[${MODULE_EXTRA}]" --target "${LAMBDA_TASK_ROOT}" -c /tmp/constraints.txt

# Install additional runtime dependencies
# UPDATE THIS PATH: examples/{module}/aws-lambda/requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy Lambda handler and business logic
# UPDATE THIS PATH: examples/{module}/aws-lambda/handler.py examples/{module}/aws-lambda/run.py
COPY handler.py run.py ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["handler.lambda_handler"]
