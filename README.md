# Causal Discovery on Higher-Order Interactions

To reproduce the results, follow the instructions below:

    - Sample the data,
    - Bootstrap the DAGs,
    - Aggregate the DAGs,
    - Plot the results using the `plot.ipynb` notebook.

## Sampling

    apt-get update && apt-get install -y graphviz r-base
    R -e 'install.packages("bnlearn")' && Rscript sample.R

## Bootstrapping

    docker build --progress=plain \
        -t bootstrapping:latest \
        -f bootstrapping/Dockerfile .
    docker run -d \
        -v $(pwd)/data:/workspace/data \
        -v $(pwd)/results:/workspace/results \
        bootstrapping:latest

## Aggregating

Replace the `YYYYMMDD-HHMMSS` string with the actual timestamp of bootstrapping
in the `results` directory to aggregate the recovered models.

    docker build --progress=plain \
        -t aggregating:latest \
        -f aggregating/Dockerfile .
    docker run -d \
        -v $(pwd)/data:/workspace/data \
        -v $(pwd)/results:/workspace/results \
        -e BOOTSTRAPPING=/workspace/results/YYYYMMDD-HHMMSS \
        aggregating:latest
