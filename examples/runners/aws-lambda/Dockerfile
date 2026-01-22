# Fair-Forge Lambda Container
# Builds from local wheel (fair-forge is not available on PyPI)

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
RUN pip install "numpy==1.26.4" "scipy==1.14.1" --target "${LAMBDA_TASK_ROOT}"

# Create constraints file to prevent scipy/numpy from being upgraded
RUN echo "numpy==1.26.4" > /tmp/constraints.txt && \
    echo "scipy==1.14.1" >> /tmp/constraints.txt

# Install the wheel with the specified extras (MODULE_EXTRA build arg)
ARG MODULE_EXTRA=runners
RUN WHEEL=$(ls /tmp/*.whl) && pip install "${WHEEL}[${MODULE_EXTRA}]" --target "${LAMBDA_TASK_ROOT}" -c /tmp/constraints.txt

# Install additional runtime dependencies
COPY examples/runners/aws-lambda/requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy Lambda handler and business logic
COPY examples/runners/aws-lambda/handler.py examples/runners/aws-lambda/run.py ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["handler.lambda_handler"]
