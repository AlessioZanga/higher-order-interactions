use causal_hub::{graphs::algorithms::metrics::shd as SHD, polars::prelude::*, prelude::*};
use chrono::Local;
use itertools::Itertools;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

fn main() {
    // Set results path.
    std::fs::create_dir_all("./results").unwrap();
    // Set run path, with bootstrapping subpath.
    let run_path = format!(
        "./results/{}/bootstrapping",
        Local::now().format("%Y%m%d-%H%M%S")
    );
    std::fs::create_dir_all(&run_path)
        .unwrap_or_else(|_| panic!("Failed to create run path '{run_path}'"));
    // Set csv path.
    let csv_path = format!("{run_path}/csv");
    std::fs::create_dir_all(&csv_path)
        .unwrap_or_else(|_| panic!("Failed to create csv path '{csv_path}'"));
    // Set dot path.
    let dot_path = format!("{run_path}/dot");
    std::fs::create_dir_all(&dot_path)
        .unwrap_or_else(|_| panic!("Failed to create dot path '{dot_path}'"));

    // Set sample size.
    let n = 100;
    // Set sample ratios.
    let k = ["0.1", "0.2", "0.5", "1.0", "2.0", "5.0"];

    // Print status.
    println!("Loading true BNs form BIF files ...");
    // Load true BNs.
    let true_bs: Vec<_> = std::fs::read_dir("./data")
        .unwrap()
        .map(|x| x.unwrap().path())
        // Filter for BIF files.
        .filter(|x| x.extension().unwrap() == "bif")
        .sorted()
        // Accumulate for later reference.
        .collect_vec()
        .into_par_iter()
        .map(|true_b| {
            // Get file prefix.
            let prefix = true_b.file_stem().unwrap().to_str().unwrap().to_owned();
            // Parse BNs from BIF file.
            let true_b = CategoricalBN::from(BIF::read(&true_b).unwrap());

            (prefix, true_b)
        })
        .collect();

    // Print status.
    println!("Loading datasets form CSV files ...");
    // Load data.
    let data: Vec<_> = true_bs
        .iter()
        .flat_map(|(prefix, true_b)| {
            k.iter()
                .map(move |k| {
                    // Specify dtypes.
                    let dtypes = vec![DataType::Utf8; true_b.graph().order()];

                    // Read train dataset.
                    let train = format!("./data/{prefix}-k_{k}-train.csv");
                    let train = CsvReader::from_path(train)
                        .unwrap()
                        .with_dtypes_slice(Some(&dtypes))
                        .low_memory(true)
                        .finish()
                        .unwrap();
                    // Cast to dataset.
                    let train: CategoricalDataSet = train.into();
                    // Align states to account for unobserved ones.
                    let train = train.with_states(
                        true_b
                            .parameters()
                            .iter()
                            .flat_map(|(_, p)| p.states().iter()),
                    );

                    // Read test dataset.
                    let test = format!("./data/{prefix}-k_{k}-test.csv");
                    let test = CsvReader::from_path(test)
                        .unwrap()
                        .with_dtypes_slice(Some(&dtypes))
                        .low_memory(true)
                        .finish()
                        .unwrap();
                    // Cast to dataset.
                    let test: CategoricalDataSet = test.into();
                    // Align states to account for unobserved ones.
                    let test = test.with_states(
                        true_b
                            .parameters()
                            .iter()
                            .flat_map(|(_, p)| p.states().iter()),
                    );

                    (k, train, test)
                })
                .map(move |(k, train, test)| (prefix, k, true_b, train, test))
        })
        .collect();

    // Align bootstrap samples.
    let grid: Vec<_> = data
        .par_iter()
        .flat_map(|x| rayon::iter::repeatn(x, n).enumerate())
        .collect();
    // Compute total number of models.
    let c = n * k.len() * true_bs.len();
    // Seed random number generator.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    // Generate shuffling seeds.
    let seeds = (0..c).map(|_| rng.gen::<u64>()).collect_vec();

    // Print status.
    println!("Performing bootstrapping ...");
    // Perform causal discovery.
    grid.into_par_iter()
        // Zip shuffling seeds.
        .zip(seeds)
        // Perform causal discovery.
        .map(|((i, (prefix, k, true_b, train, test)), seed)| {
            // Initialize the random number generator.
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            // Resample train set.
            let train_i = train.sample_with_replacement(&mut rng, train.sample_size());

            // Initialize the scoring criterion.
            let s = BIC::new(&train_i);
            // Initialize the prior knowledge to the empty set.
            let pk = FR::new(L!(true_b.graph()), [], []);
            // Perform causal discovery.
            let pred_g: DGraph = HC::new(&s).with_shuffle(&mut rng).call(&pk);

            (i, prefix, k, true_b, pred_g, train, test)
        })
        // Compute metrics.
        .for_each(|(i, prefix, k, true_b, pred_g, train, test)| {
            // Write predicted graph to file.
            let id = format!("{i}-{prefix}-{k}-none");
            let id = format!("{:x}", md5::compute(id));
            DOT::from(pred_g.clone())
                .write(format!("{dot_path}/{id}.dot"))
                .expect("Failed to write predicted graph to file");
            // Compute confusion matrix.
            let cm = ConfusionMatrix::from((&true_b.graph().clone(), &pred_g.clone()));

            let in_bic = BIC::new(train);
            let out_bic = BIC::new(test);
            let in_bic = ScoringCriterion::call(&in_bic, &pred_g);
            let out_bic = ScoringCriterion::call(&out_bic, &pred_g);
            let shd = SHD(true_b.graph(), &pred_g);
            let sensitivity = cm.true_positive_rate();
            let specificity = cm.true_negative_rate();
            let accuracy = cm.accuracy();
            let balanced_accuracy = cm.balanced_accuracy();
            let f1 = cm.f1_score();

            // Create dataframe file.
            let mut f = std::fs::File::create(format!("{csv_path}/{id}.csv")).unwrap();
            // Save results dataframe "incrementally".
            let mut r = DataFrame::new(vec![
                Series::new("id", [id]),
                Series::new("bn", [prefix.to_string()]),
                Series::new("sample_ratio", [k.to_string()]),
                Series::new("method", ["none".to_string()]),
                Series::new("in_bic", [in_bic]),
                Series::new("out_bic", [out_bic]),
                Series::new("shd", [shd]),
                Series::new("sensitivity", [sensitivity]),
                Series::new("specificity", [specificity]),
                Series::new("accuracy", [accuracy]),
                Series::new("balanced_accuracy", [balanced_accuracy]),
                Series::new("f1", [f1]),
            ])
            .unwrap();
            // Write dataframe to file "incrementally".
            CsvWriter::new(&mut f).finish(&mut r).unwrap();
        });
}
