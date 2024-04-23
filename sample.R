library(bnlearn)

# Set seed.
set.seed(42)
# List BIF files from data directory.
files <- list.files("./data", full.names = TRUE)
files <- files[substr(files, nchar(files) - 3, nchar(files)) == ".bif"]
# Sample from BNs, sample size is computed from number of nodes.
for (file in files) {
    # Print.
    print(paste("Preprocessing:", file, "..."))
    # Load BN from BIF files.
    bn <- read.bif(file)
    # For each sample ratio.
    for (k in c(0.1, 0.2, 0.5, 1, 2, 5)) {
        # Compute sample size.
        n <- floor(k * nparams(bn))
        # Sample train and test dataset.
        train <- rbn(bn, n)
        test <- rbn(bn, n)
        # Write train and test dataset to file.
        write.csv(
            train,
            file = paste0(
                substr(file, 1, nchar(file) - 4),
                "-k_",
                format(k, nsmall = 1),
                "-train.csv"
            ),
            row.names = FALSE
        )
        write.csv(
            test,
            file = paste0(
                substr(file, 1, nchar(file) - 4),
                "-k_",
                format(k, nsmall = 1),
                "-test.csv"
            ),
            row.names = FALSE
        )
    }
}
