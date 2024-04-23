use causal_hub::{graphs::algorithms::metrics::shd as SHD, polars::prelude::*, prelude::*};
use itertools::Itertools;
use rayon::prelude::*;

/// Threshold-based Model Averaging.
pub struct TMA {
    alpha: f64,
}

impl TMA {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    pub fn par_call<'a, I, G>(&self, iter: I) -> G
    where
        I: IntoIterator<Item = &'a G>,
        G: Sync + DirectedGraph<Direction = directions::Directed> + PathGraph + 'a,
    {
        // Accumulate the iterator.
        let g = iter.into_iter().collect_vec();
        let n = g.len();
        // Get reference vertex set.
        let v = L!(g[0]).collect_vec();
        // Assert same vertex set.
        assert!(g.par_iter().all(|g| L!(g).zip(&v).all(|(x, y)| &x == y)));

        // Compute F.
        let mut f: Vec<_> = g
            .into_par_iter()
            // Compute F for each G.
            .map(|g| E!(g).map(|e| (e, 1_usize)).collect())
            // Aggregate Fs.
            .reduce(FxIndexMap::default, |mut f_a, mut f_b| {
                // Swap in order to iterate on the smaller.
                if f_a.len() > f_b.len() {
                    std::mem::swap(&mut f_a, &mut f_b);
                }
                // Compute F_b += F_a.
                f_a.into_iter().for_each(|(e, f_e)| {
                    *f_b.entry(e).or_default() += f_e;
                });

                f_b
            })
            // Normalize F / |G|.
            .into_par_iter()
            .map(|(e, f_e)| (e, f_e as f64 / n as f64))
            // Threshold F > alpha.
            .filter(|(_, f_e)| f_e > &self.alpha)
            .collect();

        // Sort F in descending order.
        f.par_sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        // Initialize average graph.
        let mut g = G::empty(v);
        // Compute average graph.
        for ((x, y), _) in f {
            // Add an edge.
            g.add_edge(x, y);
            // If it introduces a cycle ...
            if !g.is_acyclic() {
                // ... remove it.
                g.del_edge(x, y);
            }
        }

        g
    }
}

/// Parents-based Model Averaging.
pub struct PMA {
    alpha: f64,
}

impl PMA {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    pub fn par_call<'a, I, G>(&self, iter: I) -> G
    where
        I: IntoIterator<Item = &'a G>,
        G: Sync + DirectedGraph<Direction = directions::Directed> + PathGraph + 'a,
    {
        // Accumulate the iterator.
        let g = iter.into_iter().collect_vec();
        // Get reference vertex set.
        let v = L!(g[0]).collect_vec();
        // Assert same vertex set.
        assert!(g.par_iter().all(|g| L!(g).zip(&v).all(|(x, y)| &x == y)));

        // Compute F.
        let mut f: Vec<_> = g
            .into_par_iter()
            // Compute F for each G.
            .map(|g| {
                V!(g)
                    .map(|x| {
                        FxIndexMap::from_iter(
                            // Map as (Pa_X, 1).
                            [(Pa!(g, x).collect_vec(), 1_usize)],
                        )
                    })
                    .collect()
            })
            // Aggregate Fs.
            .reduce(Vec::new, |f_a, f_b| {
                // NOTE: Early return if one of the two is empty to avoid
                //       the zip operator to stop early and drop values.
                if f_a.is_empty() {
                    return f_b;
                }
                if f_b.is_empty() {
                    return f_a;
                }

                // For each vertex ...
                f_a.into_iter()
                    .zip(f_b)
                    .map(|(mut f_a, mut f_b)| {
                        // Swap in order to iterate on the smaller.
                        if f_a.len() > f_b.len() {
                            std::mem::swap(&mut f_a, &mut f_b);
                        }
                        // Compute F_b += F_a.
                        f_a.into_iter().for_each(|(pa_x, f_a)| {
                            *f_b.entry(pa_x).or_default() += f_a;
                        });

                        f_b
                    })
                    .collect()
            })
            // Normalize F[X, .] / sum F[X, Pa_X].
            .into_par_iter()
            .enumerate()
            .flat_map(|(x, pa_x)| {
                // Compute sum F[X, Pa_X].
                let s: usize = pa_x.values().sum();
                // Compute F[X, .] / sum F[X, Pa_X].
                pa_x.into_par_iter()
                    .map(move |(pa_x, f_pa_x)| (x, pa_x, f_pa_x as f64 / s as f64))
            })
            // Threshold F > alpha.
            .filter(|(_, _, f_pa_x)| f_pa_x > &self.alpha)
            .collect();

        // Sort F in descending order.
        f.par_sort_by(|(_, _, a), (_, _, b)| b.partial_cmp(a).unwrap());

        // Initialize flag for already added vertices.
        let mut a = vec![false; v.len()];
        // Initialize average graph.
        let mut g = G::empty(v);
        // Add a parent set if it does not introduce a cycle.
        for (x, pa_x, _) in f {
            // If it has not been already added.
            if !a[x] {
                // Add an edge.
                for &y in &pa_x {
                    g.add_edge(y, x);
                }
                // If it introduces a cycle ...
                if !g.is_acyclic() {
                    // ... remove it.
                    for y in pa_x {
                        g.del_edge(y, x);
                    }
                } else {
                    a[x] = true;
                }
            }
        }

        g
    }
}

/// Children-based Model Averaging.
pub struct CMA {
    alpha: f64,
}

impl CMA {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    pub fn par_call<'a, I, G>(&self, iter: I) -> G
    where
        I: IntoIterator<Item = &'a G>,
        G: Sync + DirectedGraph<Direction = directions::Directed> + PathGraph + 'a,
    {
        // Accumulate the iterator.
        let g = iter.into_iter().collect_vec();
        // Get reference vertex set.
        let v = L!(g[0]).collect_vec();
        // Assert same vertex set.
        assert!(g.par_iter().all(|g| L!(g).zip(&v).all(|(x, y)| &x == y)));

        // Compute F.
        let mut f: Vec<_> = g
            .into_par_iter()
            // Compute F for each G.
            .map(|g| {
                V!(g)
                    .map(|x| {
                        FxIndexMap::from_iter(
                            // Map as (Ch_X, 1).
                            [(Ch!(g, x).collect_vec(), 1_usize)],
                        )
                    })
                    .collect()
            })
            // Aggregate Fs.
            .reduce(Vec::new, |f_a, f_b| {
                // NOTE: Early return if one of the two is empty to avoid
                //       the zip operator to stop early and drop values.
                if f_a.is_empty() {
                    return f_b;
                }
                if f_b.is_empty() {
                    return f_a;
                }

                // For each vertex ...
                f_a.into_iter()
                    .zip(f_b)
                    .map(|(mut f_a, mut f_b)| {
                        // Swap in order to iterate on the smaller.
                        if f_a.len() > f_b.len() {
                            std::mem::swap(&mut f_a, &mut f_b);
                        }
                        // Compute F_b += F_a.
                        f_a.into_iter().for_each(|(ch_x, f_a)| {
                            *f_b.entry(ch_x).or_default() += f_a;
                        });

                        f_b
                    })
                    .collect()
            })
            // Normalize F[X, .] / sum F[X, Ch_X].
            .into_par_iter()
            .enumerate()
            .flat_map(|(x, ch_x)| {
                // Compute sum F[X, Ch_X].
                let s: usize = ch_x.values().sum();
                // Compute F[X, .] / sum F[X, Ch_X].
                ch_x.into_par_iter()
                    .map(move |(ch_x, f_ch_x)| (x, ch_x, f_ch_x as f64 / s as f64))
            })
            // Threshold F > alpha.
            .filter(|(_, _, f_ch_x)| f_ch_x > &self.alpha)
            .collect();

        // Sort F in descending order.
        f.par_sort_by(|(_, _, a), (_, _, b)| b.partial_cmp(a).unwrap());

        // Initialize flag for already added vertices.
        let mut a = vec![false; v.len()];
        // Initialize average graph.
        let mut g = G::empty(v);
        // Add a children set if it does not introduce a cycle.
        for (x, ch_x, _) in f {
            // If it has not been already added.
            if !a[x] {
                // Add an edge.
                for &y in &ch_x {
                    g.add_edge(x, y);
                }
                // If it introduces a cycle ...
                if !g.is_acyclic() {
                    // ... remove it.
                    for y in ch_x {
                        g.del_edge(x, y);
                    }
                } else {
                    a[x] = true;
                }
            }
        }

        g
    }
}

/// Incident-based Model Averaging.
pub struct IMA {
    alpha: f64,
}

impl IMA {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    pub fn par_call<'a, I, G>(&self, iter: I) -> G
    where
        I: IntoIterator<Item = &'a G>,
        G: Sync + DirectedGraph<Direction = directions::Directed> + PathGraph + 'a,
    {
        // Accumulate the iterator.
        let g = iter.into_iter().collect_vec();
        // Get reference vertex set.
        let v = L!(g[0]).collect_vec();
        // Assert same vertex set.
        assert!(g.par_iter().all(|g| L!(g).zip(&v).all(|(x, y)| &x == y)));

        // Compute F.
        let mut f: Vec<_> = g
            .into_par_iter()
            // Compute F for each G.
            .map(|g| {
                V!(g)
                    .map(|x| {
                        FxIndexMap::from_iter(
                            // Map as (Ch_X, 1).
                            [(
                                Pa!(g, x)
                                    .map(|z| (z, x))
                                    .chain(Ch!(g, x).map(|y| (x, y)))
                                    .collect_vec(),
                                1_usize,
                            )],
                        )
                    })
                    .collect()
            })
            // Aggregate Fs.
            .reduce(Vec::new, |f_a, f_b| {
                // NOTE: Early return if one of the two is empty to avoid
                //       the zip operator to stop early and drop values.
                if f_a.is_empty() {
                    return f_b;
                }
                if f_b.is_empty() {
                    return f_a;
                }

                // For each vertex ...
                f_a.into_iter()
                    .zip(f_b)
                    .map(|(mut f_a, mut f_b)| {
                        // Swap in order to iterate on the smaller.
                        if f_a.len() > f_b.len() {
                            std::mem::swap(&mut f_a, &mut f_b);
                        }
                        // Compute F_b += F_a.
                        f_a.into_iter().for_each(|(e_x, f_a)| {
                            *f_b.entry(e_x).or_default() += f_a;
                        });

                        f_b
                    })
                    .collect()
            })
            // Normalize F[X, .] / sum F[X, e_X].
            .into_par_iter()
            .enumerate()
            .flat_map(|(x, e_x)| {
                // Compute sum F[X, e_x].
                let s: usize = e_x.values().sum();
                // Compute F[X, .] / sum F[X, e_x].
                e_x.into_par_iter()
                    .map(move |(e_x, f_e_x)| (x, e_x, f_e_x as f64 / s as f64))
            })
            // Threshold F > alpha.
            .filter(|(_, _, f_e_x)| f_e_x > &self.alpha)
            .collect();

        // Sort F in descending order.
        f.par_sort_by(|(_, _, a), (_, _, b)| b.partial_cmp(a).unwrap());

        // Initialize flag for already added vertices.
        let mut a = vec![false; v.len()];
        // Initialize average graph.
        let mut g = G::empty(v);
        // Add a children set if it does not introduce a cycle.
        for (x, e_x, _) in f {
            // If it has not been already added.
            if !a[x] {
                // Add an edge.
                for &(z, y) in &e_x {
                    g.add_edge(z, y);
                }
                // If it introduces a cycle ...
                if !g.is_acyclic() {
                    // ... remove it.
                    for (z, y) in e_x {
                        g.del_edge(z, y);
                    }
                } else {
                    a[x] = true;
                }
            }
        }

        g
    }
}

fn main() {
    // Set results path.
    let args: Vec<String> = std::env::args().collect();
    let run_path = &args[1];
    // Set run paths, with aggregating and bootstrapping subpaths.
    let aggr_path = format!("{}/aggregating", run_path);
    let boot_path = format!("{}/bootstrapping", run_path);
    std::fs::create_dir_all(&aggr_path)
        .unwrap_or_else(|_| panic!("Failed to create run path '{aggr_path}'"));
    // Set csv paths.
    let aggr_csv_path = format!("{aggr_path}/csv");
    let boot_csv_path = format!("{boot_path}/csv");
    std::fs::create_dir_all(&aggr_csv_path)
        .unwrap_or_else(|_| panic!("Failed to create csv path '{aggr_path}'"));
    // Set dot paths.
    let aggr_dot_path = format!("{aggr_path}/dot");
    let boot_dot_path = format!("{boot_path}/dot");
    std::fs::create_dir_all(&aggr_dot_path)
        .unwrap_or_else(|_| panic!("Failed to create dot path '{aggr_path}'"));

    // Print status.
    println!("Loading pred. Gs from CSV files ...");
    // Load index shards.
    let index = std::fs::read_dir(boot_csv_path)
        .unwrap()
        .par_bridge()
        .map(|x| x.unwrap().path())
        .map(|x| {
            // Specify dtypes.
            let dtypes = vec![DataType::Utf8; 12];
            // Read shard.
            CsvReader::from_path(x)
                .unwrap()
                .with_dtypes_slice(Some(&dtypes))
                .finish()
                .unwrap()
        })
        // Merge shards into one.
        .reduce_with(|a, b| a.vstack(&b).unwrap())
        .unwrap()
        .sort(vec!["bn", "sample_ratio"], vec![false, false], true)
        .unwrap();

    // Set sample ratios.
    let k = ["0.1", "0.2", "0.5", "1.0", "2.0", "5.0"];

    // Print status.
    println!("Loading true BNs form BIF files ...");
    // Load true BNs.
    let true_bs: FxIndexMap<_, _> = std::fs::read_dir("./data")
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
    let data: FxIndexMap<_, _> = true_bs
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
                .flat_map(move |(k, train, test)| {
                    [
                        (format!("{prefix}-k_{k}-train"), train),
                        (format!("{prefix}-k_{k}-test"), test),
                    ]
                })
        })
        .collect();

    // Group by BN and sample ratio.
    let groups: Vec<_> = index
        .group_by_stable(vec!["bn", "sample_ratio"])
        .unwrap()
        // Get groups and take the underlying data frames.
        .get_groups()
        .iter()
        .map(|g| match g {
            GroupsIndicator::Idx(i) => index
                .take(&IdxCa::from_vec("", i.1.iter().copied().collect_vec()))
                .unwrap(),
            GroupsIndicator::Slice([i, n]) => index.slice(i as i64, n as usize),
        })
        .collect();

    // Print status.
    println!("Aggregate each group ...");
    // Aggregate for each group.
    groups
        .into_iter()
        .map(|g| {
            // Get associated data sets.
            let (prefix, k) = (
                g.column("bn")
                    .unwrap()
                    .utf8()
                    .unwrap()
                    .into_iter()
                    .next()
                    .unwrap()
                    .unwrap()
                    .to_owned(),
                g.column("sample_ratio")
                    .unwrap()
                    .utf8()
                    .unwrap()
                    .into_iter()
                    .next()
                    .unwrap()
                    .unwrap()
                    .to_owned(),
            );
            let (train, test) = (
                format!("{prefix}-k_{k}-train"),
                format!("{prefix}-k_{k}-test"),
            );
            let (true_b, train, test) = (&true_bs[&prefix], &data[&train], &data[&test]);
            // Load graphs.
            let pred_gs = g
                .column("id")
                .unwrap()
                .utf8()
                .unwrap()
                .into_iter()
                .map(|g| format!("{}/{}.dot", boot_dot_path, g.unwrap()))
                .map(|g| DOT::read(g).unwrap())
                .map(DGraph::from)
                .collect_vec();

            (prefix, k, true_b, pred_gs, train, test)
        })
        // Aggregate graphs.
        .map(|(prefix, k, true_b, pred_gs, train, test)| {
            // Compute potential parents threshold.
            let pma_alpha = 1. / (true_b.graph().order() - 1) as f64;
            // Apply aggregation method.
            let pred_gs = [
                ("tma_0.50", TMA::new(0.50).par_call(&pred_gs)),
                ("pma", PMA::new(pma_alpha).par_call(&pred_gs)),
                ("cma", CMA::new(pma_alpha).par_call(&pred_gs)),
                ("ima", IMA::new(pma_alpha).par_call(&pred_gs)),
            ];

            (prefix, k, true_b, pred_gs, train, test)
        })
        // Evaluate each average graph.
        .for_each(|(prefix, k, true_b, pred_gs, train, test)| {
            pred_gs.into_iter().for_each(|(method, pred_g)| {
                // Write predicted graph to file.
                let id = format!("0-{prefix}-{k}-{method}");
                let id = format!("{:x}", md5::compute(id));
                DOT::from(pred_g.clone())
                    .write(format!("{aggr_dot_path}/{id}.dot"))
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
                let mut f = std::fs::File::create(format!("{aggr_csv_path}/{id}.csv")).unwrap();
                // Save results dataframe "incrementally".
                let mut r = DataFrame::new(vec![
                    Series::new("id", [id]),
                    Series::new("bn", [prefix.to_string()]),
                    Series::new("sample_ratio", [k.to_string()]),
                    Series::new("method", [method.to_string()]),
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
        });
}
