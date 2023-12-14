FROM tensorflow/tensorflow:2.13.0-gpu

RUN groupadd --gid 5000 tfuser \
    && useradd --home-dir /home/tfuser --create-home --uid 5000 \
    --gid 5000 --shell /bin/bash --skel /dev/null tfuser

ENV SHELL /bin/bash