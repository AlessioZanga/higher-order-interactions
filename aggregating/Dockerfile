FROM rust:1.70-slim

WORKDIR /workspace

COPY . .

ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
ENV BOOTSTRAPPING=/tmp

RUN apt-get update && apt-get install -y libopenblas-dev time
RUN mkdir -p results && \
    RUSTFLAGS="-C target-cpu=native" \
    cargo build --release \
    --bin aggregating

CMD [ "sh", "-c", "time -p target/release/aggregating $BOOTSTRAPPING" ]
