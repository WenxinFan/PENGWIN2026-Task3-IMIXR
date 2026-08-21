FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN python -m pip install --no-cache-dir \
    numpy==2.1.1 \
    scipy==1.14.1 \
    trimesh==4.8.2

WORKDIR /opt/app
COPY models /opt/app/models
COPY submission_runtime/pengwin_task3 /opt/app/pengwin_task3
COPY weights/model_state_dict.pt /opt/app/model_state_dict.pt

RUN useradd --create-home --uid 1000 algorithm \
    && mkdir -p /input /output /tmp/matplotlib \
    && chown -R algorithm:algorithm /opt/app /input /output /tmp/matplotlib

USER algorithm
ENTRYPOINT ["python", "-m", "pengwin_task3"]
CMD ["--input", "/input", "--output", "/output/reduction-poses-matrices.json"]
