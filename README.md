# Pipeline

Pipeline is a collection of scripts for acquiring ENCODE data, preparing genome
and signal inputs, building experiment datasets, fitting models, and interpreting
their predictions.

Run commands from the repository root with the Python environment that provides
the project’s `poseigen_*` packages and scientific Python dependencies.
`data_prep.py` uses MACS3 on `PATH` only when single-ended BAMs need fragment
length predictions and no precomputed values are supplied.

## Workflow

Start with one JSON experiment definition containing the raw ENCODE accessions.
`data_prep.py` retrieves metadata for listed accessions, validates the target/
control relationships, and writes the generated signal IDs back to the file
before building the experiment dataset.

1. **Create the raw-accession experiment JSON.** Each file describes one
    experiment. `input_accessions` and `output_accessions` are lists of target
    ENCODE file accessions; `control_accessions` explicitly lists the control
    files for those target datasets. Every target dataset’s controls must be
    listed, and each target/control dataset needs at least two replicate files.
    The output accessions must resolve to one target/control pair. The driver
    does not download unlisted controls. When a non-control ENCODE dataset has
    no target, its assay name is used as the target label; control datasets
    remain unlabelled.

    Example format:

    ```json
    {
      "experiment_name": "EnhAct",
      "genome": "hg38",
      "input_accessions": [],
      "output_accessions": ["ENCFF000ABC", "ENCFF000ABD"],
      "control_accessions": ["ENCFF000XYZ", "ENCFF000XYA"],
      "output_size": 50,
     "seq_size": 1000,
     "input_size": 200,
      "translate_bp": 250,
      "num_trans_wins": 5,
      "unique_sites": 30000,
      "target_percentile": 1,
      "control_percentile": 5
    }
    ```

    `input_accessions` may be empty. After replicate processing,
    `data_prep.py` retains the raw accession lists and fills `input_signals`
    with generated pair IDs and `output_signal` with the generated output pair
    ID. `num_trans_wins` is optional and defaults to 3; if supplied, it must be
    positive. Sizes and `unique_sites` must be positive, while `translate_bp`
    may be zero. The optional `target_percentile` and `control_percentile`
    settings control the lower signal filters and default to 1 and 5,
    respectively; each must be between 0 and 100.

2. **[`code/data_prep.py`](code/data_prep.py) — prepare all data for the experiment.**
     Reads the raw accession lists, downloads only listed files that are missing,
     retrieves their ENCODE metadata, verifies control relationships, prepares
     missing genome resources, predicts fragment lengths only for single-ended
     BAMs that need them, runs per-file current and replicate processing, then
     writes generated signal IDs to the JSON and runs site selection, pack
     building, and split generation as separate stages.
     Per-file processing skips output files that are already present.

    Replicate processing saves per-pair VST diagnostic figures under
    `data/ENCODE/vst/figs/`: the raw-mode-to-VST curve and the local scale
    implied by that curve, plus a before/after spread comparison from observed
    target/control replicate ratios. The spread comparison uses a Harpoon-
    balanced sample of eligible genomic sites and caches its sampled statistics.
    These figures are refreshed when a cached transform is reused, without
    refitting it.

    Target-replicate concordance is reported under
    `data/ENCODE/replicate_qc/`: pairwise log1p signal plots and a JSON summary
    of correlations and zero-versus-positive disagreement. It uses a fixed
    sample of unblacklisted bins after scaling each target replicate to the
    mean total reads across its paired control replicates. The selected
    experiment's pair-specific plots and JSON summaries are also copied into
    its `output/figures/data_preparation/replicate_qc/` and
    `output/tables/data_preparation/replicate_qc/` folders.

    For hg38, the script uses the standard ENCODE GRCh38 FASTA URL automatically
    when resources are missing. For another genome or an alternate hg38 assembly,
    provide a FASTA URL with `--genome-url`:

    ```bash
    python code/data_prep.py data/experiments.json
    ```

    `--genome-url` is also unnecessary when the experiment’s genome resources
    are already present. To use precomputed fragment lengths instead of MACS3,
    pass `--predictd-file yy1_predictd.txt`. The experiment JSON’s genome label
    must match the files prepared under `data/genomes/<genome-name>/`.

    Flag generation and site selection are isolated in
    [`code/data_preparation/site_selection.py`](code/data_preparation/site_selection.py)
    and run after replicate processing, before pack generation. The script adds
    a pseudocount, flags 2,000-bp windows at 50-bp resolution, filters candidate
    sites using control thresholds and the genome blacklist, then applies the
    experiment’s VST and Harpoon sampling across 1,000 value bins. It saves the
    flag intermediates, selected sites, thresholds, and diagnostic figures
    under the experiment’s `flag/` directory. Its site-selection figure and QC
    JSON are written to `output/figures/data_preparation/site_selection/` and
    `output/tables/data_preparation/site_selection/`. Run it directly with an
    experiment JSON when only site selection needs to be regenerated. The QC
    report describes retained candidate counts after
    each complete-window blacklist, target, and control filter.

    [`code/data_preparation/pack_builder.py`](code/data_preparation/pack_builder.py)
    consumes those selected sites and creates the sequence, signal, and
    translation-augmented packs. It does not rerun site selection or generate
    splits. `data_prep.py` runs site selection, pack building, and split
    generation as separate stages; when running stages manually, run
    `site_selection.py`, then `pack_builder.py`, then `data_split.py`.
    Pack-distribution figures are written to
    `output/figures/data_preparation/pack_builder/`.

    The lower signal bounds are configurable in the experiment JSON with
    `control_percentile` (default 5) and `target_percentile` (default 1).
    Both are applied to positive finite values; the upper bounds remain the
    maximum positive value.

    The final split stage is isolated in
    [`code/data_preparation/data_split.py`](code/data_preparation/data_split.py).
    It assigns overlap groups to selected sequence windows by chromosome
    before generating the grouped splits, keeping overlapping windows together.
    It saves column-wise and row-wise normalized bin-proportion heatmaps and
    the output-value distribution under
    `output/figures/data_preparation/split/`. `data_prep.py` runs it
    automatically after pack generation. Existing `split.p` partitions are
    reused by default; their figures are refreshed without rerunning the split
    search. To explicitly generate new splits, pass `--redo-splits`:

    ```bash
    python code/data_preparation/data_split.py data/experiments.json --redo-splits
    # Or run the full preparation driver:
    python code/data_prep.py data/experiments.json --redo-splits
    ```

    The standalone script accepts `--output-root`, `--split-candidates`,
    `--split-cycles`, and `--seed`. `data_prep.py` also accepts the split search
    options; candidate, cycle, and seed options apply only when
    `--redo-splits` is set.

3. **[`code/model_search.py`](code/model_search.py) — search model configurations.**
    Searches the applicable model families using the experiment packs and splits;
    stores results and selected configurations under the experiment’s `search/`
    directory. Set `models_to_search` in the experiment JSON to a non-empty list
    of model names to run only those models; omit it to search all applicable
    models. For example, the EnhAct experiment searches only `Oyster`.

    Single-input model names are `DeepSTARR`, `Mimic_Att`, `Mimic_Single`,
    `Mimic_Recurrent`, `Mimic_CPM`, and `Oyster`. Dual-input names use the
    `Dual_` variants, with `BiChrom` in place of `DeepSTARR`.

4. **[`code/model_mimic_comps.py`](code/model_mimic_comps.py) — compare
    optimization results.** Bootstraps and summarizes candidate ranks and
    parameter preferences from the optimization output. Writes comparison CSVs
    under `output/tables/model_search/`. Run after searching the Mimic model
    families.

5. **[`code/model_fit_exp.py`](code/model_fit_exp.py) — run model-fitting
    experiments.** Uses the selected configurations to run the experiment
    fitting/replicate workflows (including the `xp2`, `xp3`, and `xp3B` runs).
    It requires a target key via `--targx`; outputs are organized under
    `data_ml/`, `search/`, and `xps/` within the experiment directory. It uses
    the selected top-refined models from `models_to_search`. Each XP summary
    also saves significance-highlighted figures under
    `output/figures/model_fitting/` and HTML/XLSX tables under
    `output/tables/model_fitting/`; bold cells indicate significant differences
    from the reference.

    ```bash
    python code/model_fit_exp.py data/experiments.json --targx K562
    ```

6. **[`code/oyster_analysis/model_selection.py`](code/oyster_analysis/model_selection.py)
    — select the fit recipe for interpretation.** Compares the existing
    bootstrap RMSE distributions for `xp2`, `xp3`, and `xp3B` against the
    `MSE-R` default. It keeps `MSE-R` unless an alternative's 5th–95th
    percentile relative-RMSE interval is entirely below zero; if several
    qualify, it chooses the one with the lowest median RMSE. The per-experiment
    selection is saved to `output/tables/model_selection.json`, and the
    repository-root `model_selection_tracker.csv` tracks the latest selection
    for each experiment.

    ```bash
    python code/oyster_analysis/model_selection.py data/experiments.json
    ```

    This uses the existing repeat bootstrap and the evaluation split; it does
    not bootstrap sites. Complete model fitting first.

7. **[`code/oyster_analysis/ensemble_analysis.py`](code/oyster_analysis/ensemble_analysis.py)
    — build an interpretation ensemble.** Uses the selected fit recipe unless
    `--xp-run` and `--loss` explicitly override it, weights fitted repeats, and
    saves the final weighted model checkpoints and their manifest under
    `output/oyster_models/<Oyster model identifier>/`. The manifest records
    selected repeat weights and relative source-fit metadata, and
    `model_config.json` preserves the selected model configuration. The
    readable weight table is saved under `output/tables/oyster_models/`.

8. **[`code/oyster_analysis/dna_analysis.py`](code/oyster_analysis/dna_analysis.py)
    — interpret DNA motifs.** Scores a MEME-format motif database against the
    DNA branch of the weighted Oyster ensemble and writes contribution tables
    and figures under `output/tables/oyster_models/` and
    `output/figures/oyster_models/`. Reusable analysis caches stay alongside
    the model metadata. Supply the motif database with `--motif-file`.

9. **[`code/oyster_analysis/signals_analysis.py`](code/oyster_analysis/signals_analysis.py)
    — interpret input signals (when applicable).** For experiments with input
    signals, analyzes the Dual_Oyster model’s DNA and signal contributions,
    including signal k-mers and clusters. It writes figures and tables under
    `output/figures/oyster_models/` and `output/tables/oyster_models/`, with
    reusable analysis caches alongside the model metadata. This step does not
    apply to experiments with no input signals.

The modeling and interpretation scripts accept the experiment JSON as an
optional positional argument and default to `data/experiments.json`. Use
`python code/data_prep.py --help` or
`python code/oyster_analysis/<script>.py --help` for interpretation options.
The individual `code/data_preparation/` scripts remain available for
standalone use.

### Genome resources

`prepare_genome.py` writes resources to `data/genomes/<genome-name>/`.
`generate_encode_currents.py`, `merge_replicates.py`, `site_selection.py`, and
`pack_builder.py` use the experiment’s genome label. The required files
include `<genome-name>_id_sizes.p`, `<genome-name>_basic.p`,
`<genome-name>_basic_id_sizes.p`, and
`<genome-name>_basic_BLC_LR50.p`.

## Repository layout

```text
.
├── README.md
├── .gitignore
├── yy1_download.txt              # ENCODE file accessions
├── yy1_predictd.txt              # ENCODE accession / fragment-length pairs
├── code/
│   ├── *.py                      # Download, processing, modeling, interpretation
│   ├── data_preparation/         # Data acquisition and preparation scripts
│   └── model_fit_experiments/    # Fitting experiment runners and shared helpers
└── data/                         # Local inputs, caches, and generated outputs
    ├── ENCODE/
    │   ├── meta_info.txt
    │   ├── raw/
    │   ├── full/ lowres/ autos/ win/
    │   └── merged/ shrunk_std/ betaprime/ vst/
    ├── genomes/<genome-name>/
    ├── experiments.json
    └── experiments/<experiment-name>/
        ├── flag/ packs/ split/
        ├── search/
        ├── data_ml/
        ├── xps/
        └── output/
            ├── experiment.json
            ├── figures/
            ├── tables/
            └── oyster_models/
```

Downloaded datasets and generated intermediates are ignored by Git.
Experiment `output/` folders are explicitly unignored so their figures, tables,
model checkpoints, and metadata can be committed and shared. The default
`data/experiments.json` is local as well; keep experiment definitions elsewhere
if they need to be version-controlled or shared.
