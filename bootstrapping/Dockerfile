FROM rust:1.70-slim

WORKDIR /workspace

COPY . .

ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

RUN apt-get update && apt-get install -y libopenblas-dev time
RUN mkdir -p results && \
    RUSTFLAGS="-C target-cpu=native" \
    cargo build --release \
    --bin bootstrapping

CMD [ "time", "-p", "target/release/bootstrapping" ]
