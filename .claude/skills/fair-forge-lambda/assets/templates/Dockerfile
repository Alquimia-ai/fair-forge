# Fair-Forge Lambda Container
# Builds from local wheel (fair-forge is not available on PyPI)

FROM public.ecr.aws/lambda/python:3.11 AS builder

# Install build tools
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

# Install the wheel with the specified extras (MODULE_EXTRA build arg)
ARG MODULE_EXTRA=bestof
RUN pip install "/tmp/$(ls /tmp/*.whl)[${MODULE_EXTRA}]" --target "${LAMBDA_TASK_ROOT}"

# Install additional runtime dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy Lambda handler and business logic
COPY handler.py run.py ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["handler.lambda_handler"]
