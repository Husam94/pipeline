# Pipeline

Pipeline is a collection of scripts for acquiring ENCODE data, preparing genome
and signal inputs, building experiment datasets, fitting models, and interpreting
their predictions.

Run commands from the repository root with the Python environment that provides
the project’s `poseigen_*` packages and scientific Python dependencies.
`predictd.py` also requires MACS3 on `PATH` when calculating fragment lengths
directly.

## Workflow

The scripts below follow the analysis lifecycle. `new_experiment.py` validates
signal IDs against `data/ENCODE/meta_info.txt`; on a fresh checkout, first run
`download_encode.py` for the ENCODE accessions you need (or otherwise provide a
metadata file) before defining the experiment.

1. **[`code/new_experiment.py`](code/new_experiment.py) — define an experiment.**
   Creates one JSON experiment object, by default at `data/experiments.json`.
   It records the experiment name, genome label, optional input signals, output
   signal, and processing parameters. Signal IDs must be file names, replicate
   IDs, or pair IDs present in `meta_info.txt`. Each JSON file describes one
   experiment; use `--output-file` for another file rather than trying to append
   a second experiment to the default file.

   Example format:

   ```json
   {
     "experiment_name": "experiment-name",
     "genome": "hg38",
     "input_signals": [],
     "output_signal": "ENCFF000ABC",
     "output_size": 100,
     "seq_size": 1000,
     "input_size": 200,
     "translate_bp": 0,
     "unique_sites": 10000
   }
   ```

   Replace the example output ID with an ID from the metadata file. Input
   signals may be an empty list. The size and `unique_sites` parameters must be
   positive integers; `translate_bp` may be zero. Parameters can also be
   supplied or overridden when running `processing_exp_data.py`.

2. **[`code/download_encode.py`](code/download_encode.py) — acquire ENCODE files
   and metadata.** Downloads the requested ENCODE files into
   `data/ENCODE/raw/` and maintains the tab-delimited
   `data/ENCODE/meta_info.txt` catalog used by later steps. For the accession
   list in this checkout, run:

   ```bash
   python code/download_encode.py --input-file yy1_download.txt
   ```

3. **[`code/prepare_genome.py`](code/prepare_genome.py) — prepare genome
   resources.** Downloads a FASTA (plain or gzip), stores genome sequence and
   chromosome-size pickles, and downloads/processes a matching Boyle-Lab
   blacklist when one is available. Outputs go to
   `data/genomes/<genome-name>/`.

   ```bash
   python code/prepare_genome.py <FASTA_URL> hg38
   ```

4. **[`code/predictd.py`](code/predictd.py) — get fragment lengths.** For
   single-ended ENCODE files, runs MACS3 `predictd` and records predicted
   fragment lengths in `meta_info.txt`. It can instead import precomputed
   accession/length pairs; `yy1_predictd.txt` is an example:

   ```bash
   python code/predictd.py --precomputed-file yy1_predictd.txt
   ```

5. **[`code/processing_initial.py`](code/processing_initial.py) — make
   per-file signal currents.** Converts BAM or bigWig inputs to full,
   low-resolution, autosomal, and windowed, blacklist-filtered currents under
   `data/ENCODE/{full,lowres,autos,win}/`. Single-ended BAMs need a `predictd`
   value in the metadata.

6. **[`code/processing_replicates.py`](code/processing_replicates.py) — merge
   replicates and build paired signal statistics.** Uses the ENCODE metadata to
   group replicate files and target/control pairs. Writes merged currents,
   shrunk standard deviations, beta-prime summaries, and variance-stabilizing
   transforms under `data/ENCODE/{merged,shrunk_std,betaprime,vst}/`.

7. **[`code/processing_exp_data.py`](code/processing_exp_data.py) — build the
   experiment dataset.** Resolves the experiment’s signal IDs, selects and
   filters sites, creates DNA/output/input packs, and generates train/stop/eval/
   test splits. Results are written to
   `data/experiments/<experiment-name>/`, primarily in `flag/`, `packs/`, and
   `split/`. For example:

   ```bash
   python code/processing_exp_data.py data/experiments.json
   ```

8. **[`code/model_opt.py`](code/model_opt.py) — optimize model configurations.**
   Searches the applicable model families using the experiment packs and splits;
   stores optimization results and selected configurations under the
   experiment’s `opt/` directory.

9. **[`code/model_ultra_comps.py`](code/model_ultra_comps.py) — compare
   optimization results.** Bootstraps and summarizes candidate ranks and
   parameter preferences from the optimization output. Writes comparison CSVs
   under `opt/tables/`. Run after `model_opt.py`.

10. **[`code/model_fit_exp.py`](code/model_fit_exp.py) — run model-fitting
    experiments.** Uses the selected configurations to run the experiment
    fitting/replicate workflows (including the `xp2`, `xp3`, and `xp3B` runs).
    It requires a target key via `--targx`; outputs are organized under
    `data_ml/`, `opt/`, and `xps/` within the experiment directory.

    ```bash
    python code/model_fit_exp.py data/experiments.json --targx K562
    ```

11. **[`code/interp_ensemble.py`](code/interp_ensemble.py) — build an
    interpretation ensemble.** Selects and weights fitted model repeats using
    bootstrap scores, then saves an ensemble manifest and weights alongside
    the selected fit results. It requires completed optimization and fitting
    outputs.

12. **[`code/interp_dna.py`](code/interp_dna.py) — interpret DNA motifs.**
    Scores a MEME-format motif database against the DNA branch of the weighted
    Oyster ensemble and writes contribution tables and figures under the
    ensemble’s `dna/` directory. Supply the motif database with `--motif-file`.

13. **[`code/interp_signals.py`](code/interp_signals.py) — interpret input
    signals (when applicable).** For experiments with input signals, analyzes
    the Dual_Oyster model’s DNA and signal contributions, including signal
    k-mers and clusters. It writes results under the ensemble’s `signals/`
    directory. This step does not apply to experiments with no input signals.

The modeling and interpretation scripts accept the experiment JSON as an
optional positional argument and default to `data/experiments.json`. Use
`python code/<script>.py --help` for each script’s options.

### Genome resource path note

`prepare_genome.py` writes resources to `data/genomes/<genome-name>/`, which is
where `processing_exp_data.py` reads them. However, `processing_initial.py` and
`processing_replicates.py` currently expect the hg38 processing pickles under
`../../Essentials/hg38` relative to the repository root. For those steps, make
the required hg38 resources (including `hg38_id_sizes.p`,
`hg38_basic_id_sizes.p`, and `hg38_basic_BLC_LR50.p`) available at that
location, or update the scripts to use the generated genome directory.

## Repository layout

```text
.
├── README.md
├── .gitignore
├── yy1_download.txt              # ENCODE file accessions
├── yy1_predictd.txt              # ENCODE accession / fragment-length pairs
├── code/
│   ├── *.py                      # Download, processing, modeling, interpretation
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
        ├── opt/
        ├── data_ml/
        └── xps/
```

The entire `/data/` directory is ignored by Git because it contains downloaded
datasets, generated binary intermediates, models, and local experiment files.
The default `data/experiments.json` is therefore local as well; keep experiment
definitions elsewhere if they need to be version-controlled or shared.
