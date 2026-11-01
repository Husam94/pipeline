import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors

import os, glob, shutil 
from pathlib import Path

import concurrent.futures

import poseigen_seaside.basics as se
import poseigen_seaside.metrics as mex
import poseigen_binmeths.bin_utils as bu

#========================================================

def load_fit_data(data_path):
    root = Path(data_path)
    experiment_paths = (
        root / "packs" / "tri_tpacks.p",
        root / "packs" / "output_bin_sizes.p",
        root / "split" / "split_transaug.p",
    )
    if all(path.is_file() for path in experiment_paths):
        packs_path, bin_sizes_path, split_path = experiment_paths
    else:
        legacy_paths = (
            root / "data_ml" / "tri_tpacks.p",
            root / "data_ml" / "bin_sizes.p",
            root / "data_ml" / "split_transaug.p",
        )
        if not all(path.is_file() for path in legacy_paths):
            missing = [str(path) for path in experiment_paths if not path.is_file()]
            raise FileNotFoundError(
                "Could not find experiment packs and split under packs/ and "
                "split/, or legacy inputs under data_ml/. Missing: "
                + ", ".join(missing)
            )
        packs_path, bin_sizes_path, split_path = legacy_paths

    tri_tpacks = se.PickleLoad(str(packs_path))
    bin_sizes = se.PickleLoad(str(bin_sizes_path))
    split_transaug = se.PickleLoad(str(split_path))
    if not isinstance(tri_tpacks, list) or len(tri_tpacks) not in (4, 5):
        raise ValueError(
            f"Expected four or five arrays in {packs_path}, "
            f"found {len(tri_tpacks) if isinstance(tri_tpacks, list) else 'invalid data'}."
        )
    return tri_tpacks, bin_sizes, split_transaug


def XpsCopy(cur_pn, cur_xp_id, 
            past_xp_id, past_unit, 
            rewrite = False):
                         
    # Replace experiment ID
    temp_path = cur_pn.replace(cur_xp_id, past_xp_id)
    # Split path and replace last directory
    parts = temp_path.rstrip('/').split('/')
    parts[-1] = str(past_unit)
    basetargo = '/'.join(parts) + '/'

    print(basetargo)

    targnames = glob.glob(basetargo + '*')
    lbt = len(basetargo)

    targnames1 = [t for t in targnames if '.p' in t or '.pt' in t]
    newnames1 = [cur_pn + t[lbt:] for t in targnames1]

    print(targnames1, newnames1)


    if rewrite: 
        for n in newnames1: 
            if os.path.isfile(n): os.remove(n)
    for t,n in zip(targnames1, newnames1): 
        if os.path.isfile(n): break #means we already have it
        shutil.copyfile(t, n)

    targnames2 = [t for t in targnames if '.p' not in t and'.pt' not in t]
    newnames2 = [cur_pn + t[lbt:] for t in targnames2]
    if rewrite: 
        for n in newnames2: 
            if os.path.isdir(n): shutil.rmtree(n)
    for t, n in zip(targnames2, newnames2): 
        if os.path.isdir(n): break #means we already have it             
        shutil.copytree(t, n, dirs_exist_ok=True)
    
    print(f'finished copying for {cur_pn}')
    
    return

#====================================================================================================

''' FOR NON DATA SPLIT EXPERIMENTS'''

def compute_ensemble_dict(xpx_dict,
                          tri_tpacks, bin_sizes, split,
                          ensemb=50, top=3, max_workers=None):
    """
    Compute ensemble predictions for all conditions in an experiment.
    
    Args:
        xp_dict: dictionary containing all experiment configurations
        synth_geom: synthetic geometry data
        ensemb: number of ensemble iterations (default: 20)
        top: number of top models to ensemble (default: 3)
        max_workers: number of parallel workers (default: os.cpu_count())
    
    Returns:
        ensemb_dict: nested dict with ensemble scores, indices, and epochs
    """

    #======================================
    
    if len(tri_tpacks) == 4:
        d_x, d_y_ms, d_s, d_b = tri_tpacks
    else:
        d_x1, d_y_ms, d_s, d_b, d_x2 = tri_tpacks
        
    split_eval = np.hstack([split[2], split[2] + np.sum([len(x) for x in split])])

    d_y_full= np.vstack([d_y_ms, d_y_ms])
    d_y_eval = d_y_full[split_eval]

    d_b_full = np.vstack([d_b, d_b])
    d_b_eval = d_b_full[split_eval]

    d_w_eval = bu.BinWeights(d_b_eval, bin_sizes = bin_sizes, mode = 'recip')

    #======================================
    
    if max_workers is None: max_workers = os.cpu_count()
    
    # Initialize ensemble dict
    ensemb_dict = {}
    
    # Build arguments list for all combinations
    args_list = []

    for cof_name in xpx_dict['modx_names_full']:
        ensemb_dict[cof_name] = {}

        for icom, com in xpx_dict['combs']:
            args_list.append((cof_name, com, 
                                xpx_dict, d_y_eval, d_w_eval, split_eval, 
                                ensemb, top
                            ))
    
    # Process in parallel
    print(f"Processing {len(args_list)} combinations...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compute_ensemble_for_condition, args_list))
    
    # Populate ensemb_dict with results
    for result in results:
        if result is not None:
            cof_namex, comx, data = result
            ensemb_dict[cof_namex][comx] = data
    
    print("Processing complete!")
    
    # Save results
    se.PickleDump(ensemb_dict, xpx_dict['pn'] + 'ensemb_dict')
    
    return ensemb_dict



def compute_ensemble_for_condition(args):
    """Compute ensemble scores for a single condition."""
    cof_name, com, xpx_dict, d_y_eval, d_w_eval, split_eval, ensemb, top = args
    
    try:
        comx = com[0]

        pn_comx = xpx_dict['pn'] + f"{cof_name}/{comx}/"
        
        preds_dict = {rep: se.PickleLoad(pn_comx + f"0_{rep}_Preds")[split_eval]
                     for rep in np.arange(xpx_dict['TCR_repeats'])}
        
        val_scores = np.array([v for k,v in se.PickleLoad(pn_comx + 'Mets')[0].items()])
        
        epoch_dict = {rep: len(se.PickleLoad(pn_comx + f"0_{rep}_Met")['Train'])
                     for rep in np.arange(xpx_dict['TCR_repeats'])}
        
        ensemb_MSE = []
        ensemb_RMSE = []
        ensemb_corr = []
        ensemb_idxs = []
        ensemb_idxs_top = []
        ensemb_epochs = []
        
        for iex in range(ensemb):
            rand_idx = np.random.choice(np.arange(xpx_dict['TCR_repeats']), 
                                       size=xpx_dict['TCR_repeats'], 
                                       replace=True)
            ensemb_idxs.append(rand_idx)
            
            val_scores_rand = val_scores[rand_idx]
            top_rand = rand_idx[np.argsort(val_scores_rand)[:top]]

            ensemb_idxs_top.append(top_rand)
            
            epochs_ensemb = np.mean([epoch_dict[x] for x in top_rand])
            ensemb_epochs.append(epochs_ensemb)
            
            preds_ensemb = np.mean(np.array([preds_dict[x] for x in top_rand]), axis=0)

            corr =  mex.Correlation(*[x.reshape(-1) for x in [d_y_eval, preds_ensemb]], weights=d_w_eval.reshape(-1))
            ensemb_corr.append(corr)

            mse = mex.AError(preds_ensemb, d_y_eval, weights=d_w_eval, 
                              expo=2, root=False, weightsum=True)
            ensemb_MSE.append(mse)

            rmse = mex.AError(preds_ensemb, d_y_eval, weights=d_w_eval, 
                              expo=2, root=True, weightsum=True)
            ensemb_RMSE.append(rmse)
        
        return (cof_name, comx, {
            'mses': np.array(ensemb_MSE),
            'rmses': np.array(ensemb_RMSE),
            'corrs': np.array(ensemb_corr),
            'idxs': np.array(ensemb_idxs),
            'idxs_top': np.array(ensemb_idxs_top),
            'epochs': np.array(ensemb_epochs)
        })
    
    except Exception as e:
        import traceback
        print(f"Error processing {cof_name}/{comx}: {e}")
        traceback.print_exc()
        return None
    






def compute_vsref_dict(xpx_dict,
                       ensemb_dict, ref_idx=0, undersamp_props=None, 
                       pf_args=None, max_workers=None):
    """
    Compute vs-reference comparisons for all conditions in an experiment.
    
    Args:
        xp: experiment number (e.g., 2, 3, 4)
        xp_dict: dictionary containing all experiment configurations
        ensemb_dict: ensemble dictionary from compute_ensemble_dict()
        ref_idx: index of reference combination (default: 0)
        undersamp_props: list of undersampling proportions (default: None)
        pf_args: arguments for PairwiseFuncer (default: {'mode1': [se.RelativeChange, {'perc': True}]})
        max_workers: number of parallel workers (default: os.cpu_count())
    
    Returns:
        ensemb_vsref: nested dict with vs-reference comparisons
    """
    
    if max_workers is None:
        max_workers = os.cpu_count()
    
    if pf_args is None:
        pf_args = {'mode1': [se.RelativeChange, {'perc': True}]}
    
    # Get reference combination
    ref_comx = xpx_dict['combs'][ref_idx][1][0]
    
    # Initialize ensemb_vsref dict
    ensemb_vsref = {}
    for cof_name in xpx_dict['modx_names_full']:
        ensemb_vsref[cof_name] = {}
    
    # Build arguments list
    args_list = []
    for cof_name in xpx_dict['modx_names_full']:
        args_list.append((cof_name, ref_comx, ref_idx,
                          xpx_dict, ensemb_dict, undersamp_props, pf_args))
    
    # Process in parallel
    print(f"Processing {len(args_list)} conditions...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compute_vsref_for_condition, args_list))
    
    # Populate ensemb_vsref with results
    for result in results:
        if result is not None:
            cof_name, data = result
            ensemb_vsref[cof_name] = data
    
    print("Processing complete!")
    
    # Save results
    se.PickleDump(ensemb_vsref, xpx_dict['pn'] + f'ensemb_vsref_{ref_comx}')
    
    return ensemb_vsref


def compute_vsref_for_condition(args):
    """Compute vs-reference comparisons for a single condition."""
    cof_name, ref_comx, ref_idx, xpx_dict, ensemb_dict, undersamp_props, pf_args = args
    
    try:
        result = {}
        
        mses_refo = ensemb_dict[cof_name][ref_comx]['mses']
        rmses_refo = ensemb_dict[cof_name][ref_comx]['rmses']
        corrs_refo = ensemb_dict[cof_name][ref_comx]['corrs']
        us_refo = undersamp_props[ref_idx] if undersamp_props is not None else 1
        epochs_refo = ensemb_dict[cof_name][ref_comx]['epochs'] * us_refo
        
        for icom, com in xpx_dict['combs']:
            comx = com[0]
            mses_compa = ensemb_dict[cof_name][comx]['mses']
            rmses_compa = ensemb_dict[cof_name][comx]['rmses']
            corrs_compa = ensemb_dict[cof_name][comx]['corrs']
            
            us_compa = undersamp_props[icom] if undersamp_props is not None else 1
            epochs_compa = ensemb_dict[cof_name][comx]['epochs'] * us_compa
            
            result[comx] = {
                'mses': np.array(se.PairwiseFuncer(mses_compa, mses_refo, **pf_args)),
                'rmses': np.array(se.PairwiseFuncer(rmses_compa, rmses_refo, **pf_args)),
                'corrs': np.array(se.PairwiseFuncer(corrs_compa, corrs_refo, **pf_args)),
                'epochs': np.array(se.PairwiseFuncer(epochs_compa, epochs_refo, **pf_args))
            }
        
        return (cof_name, result)
    
    except Exception as e:
        import traceback
        print(f"Error processing {cof_name}: {e}")
        traceback.print_exc()
        return None



#------------------------------------------------

def compute_vsref_stats_dict(xpx_dict,
                             ensemb_vsref, ref_idx=0, 
                              percentile_low=2.5, percentile_high=97.5, max_workers=None):
    """
    Compute statistics from vs-reference comparisons for all conditions.
    
    Args:
        xp_dict: dictionary containing all experiment configurations
        ensemb_vsref: vs-reference dictionary from compute_vsref_dict()
        ref_idx: index of reference combination (default: 0)
        percentile_low: lower percentile for confidence interval (default: 2.5)
        percentile_high: upper percentile for confidence interval (default: 97.5)
        max_workers: number of parallel workers (default: os.cpu_count())
    
    Returns:
        ensemb_vsref_stats: nested dict with statistics for each condition
    """

    if max_workers is None:
        max_workers = os.cpu_count()
    
    # Get reference combination name
    ref_comx = xpx_dict['combs'][ref_idx][1][0]
    
    # Initialize ensemb_vsref_stats dict
    ensemb_vsref_stats = {}
    for cof_name in xpx_dict['modx_names_full']:
        ensemb_vsref_stats[cof_name] = {}

    
    # Build arguments list
    args_list = []
    for cof_name in xpx_dict['modx_names_full']:
        for icom, com in xpx_dict['combs']:
            comx = com[0]
            args_list.append((
                        cof_name, comx, xpx_dict, ensemb_vsref, 
                        percentile_low, percentile_high
                    ))


    # Process in parallel
    print(f"Processing {len(args_list)} conditions...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compute_stats_for_condition, args_list))
    
    # Populate ensemb_vsref_stats with results
    for result in results:
        if result is not None:
            cof_name, comx, stats = result
            ensemb_vsref_stats[cof_name][comx] = stats
    
    print("Processing complete!")
    
    # Save results
    se.PickleDump(ensemb_vsref_stats, xpx_dict['pn'] + f'ensemb_vsref_{ref_comx}_stats')
    
    return ensemb_vsref_stats


def compute_stats_for_condition(args):
    """Compute statistics for a single (pl, nx, comx) condition."""
    cof_name, comx, xpx_dict, ensemb_vsref, percentile_low, percentile_high = args
    
    try:
        mses_all = []
        rmses_all = []
        corrs_all = []
        epochs_all = []
        
        mses_all.append(ensemb_vsref[cof_name][comx]['mses'])
        rmses_all.append(ensemb_vsref[cof_name][comx]['rmses'])
        corrs_all.append(ensemb_vsref[cof_name][comx]['corrs'])
        epochs_all.append(ensemb_vsref[cof_name][comx]['epochs'])
        
        results = {}
        for subj, subj_str in zip([mses_all, rmses_all, corrs_all, epochs_all], 
                                  ['mses', 'rmses', 'corrs', 'epochs']):
            mean = np.mean(subj)
            median = np.median(subj)
            low_interval = np.percentile(subj, percentile_low)
            high_interval = np.percentile(subj, percentile_high)
            signif = 0 < low_interval or 0 > high_interval
            
            results[subj_str] = {
                'mean': mean, 
                'median': median,
                'low': low_interval, 
                'high': high_interval,
                'signif': signif
            }
        
        return (cof_name, comx, results)
    
    except Exception as e:
        import traceback
        print(f"Error processing {cof_name}/{comx}: {e}")
        traceback.print_exc()
        return None
    







    #----------------------------------

def build_vsref_stats_table(ensemb_vsref_stats, xpx_dict,
                            metric='mses', use_median=True, vmin=None, vmax=None):
    """
    Build styled pandas table from ensemb_vsref_stats.
    
    Args:
        ensemb_vsref_stats: nested dict with structure [cof_name][comx][metric]
        xp_dict: experiment dictionary containing allconfigs
        metric: 'mses', 'rmses', 'corr', or 'epochs'
        use_median: if True use 'median', else use 'mean'
        vmin, vmax: optional min/max for colormap. If None, computed from data
    
    Returns:
        styled pandas DataFrame, results DataFrame, significance DataFrame
    """
    # Use original order from xp_dict
    cof_names = xpx_dict['modx_names_full']  # Preserves order of configurations
    
    # Get comxs in order they appear in the data
    comxs = []
    if cof_names:
        first_cof = cof_names[0]
        if first_cof in ensemb_vsref_stats:
            comxs = list(ensemb_vsref_stats[first_cof].keys())
    
    # Build results and significance dicts
    results = {comx: {} for comx in comxs}
    significance = {comx: {} for comx in comxs}
    
    stat_key = 'median' if use_median else 'mean'
    
    for cof_name in cof_names:
        for comx in comxs:
            # Get value and significance
            value = ensemb_vsref_stats[cof_name][comx][metric][stat_key]
            is_signif = ensemb_vsref_stats[cof_name][comx][metric]['signif']
            
            results[comx][cof_name] = value
            significance[comx][cof_name] = is_signif
    
    # Create DataFrames with ordered index
    df_results = pd.DataFrame(results).T
    df_significance = pd.DataFrame(significance).T
    
    # Reindex to preserve comx order
    df_results = df_results.reindex(comxs)
    df_significance = df_significance.reindex(comxs)
    
    # Reorder columns to match original order
    df_results = df_results[cof_names]
    df_significance = df_significance[cof_names]
    
    # Determine vmin/vmax if not provided
    if vmin is None:
        vmin = df_results.min().min()
    if vmax is None:
        vmax = df_results.max().max()
    
    # Style function for bold significance
    def highlight_significant(data):
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        for idx in data.index:
            for col in data.columns:
                if df_significance.loc[idx, col]:  # Use .loc for explicit indexing
                    styles.loc[idx, col] = 'font-weight: bold'
        return styles
    
    # Apply styling
    styled = df_results.style\
        .format("{:.2f}")\
        .apply(highlight_significant, axis=None)\
        .background_gradient(axis=None, cmap="Greys", vmin=vmin, vmax=vmax)
    
    return styled, df_results, df_significance



#----------------------------------------

def build_full_stats_table(ensemb_vsref_stats, xpx_dict, metric='mses'):
    """
    Build a flat table of all statistics for each (cof_name, comx).

    Args:
        ensemb_vsref_stats: nested dict with structure [cof_name][comx][metric]
        xpx_dict: experiment dictionary containing allconfigs
        metric: 'mses', 'rmses', 'corrs', or 'epochs'

    Returns:
        pandas DataFrame with all statistics
    """
    cof_names = list(xpx_dict['modx_names_full'])

    # Preserve comx order from first config
    comxs = []
    if cof_names:
        first_cof = cof_names[0]
        if first_cof in ensemb_vsref_stats:
            comxs = list(ensemb_vsref_stats[first_cof].keys())

    rows = []
    for cof_name in cof_names:
        for comx in comxs:
            stats = ensemb_vsref_stats[cof_name][comx][metric]
            rows.append({
                'cof_name': cof_name,
                'comx': comx,
                'mean': stats['mean'],
                'median': stats['median'],
                'low': stats['low'],
                'high': stats['high'],
                'signif': stats['signif']
            })

    return pd.DataFrame(rows)


def build_full_stats_table_styled(ensemb_vsref_stats, xpx_dict, metric='mses',
                                  format_dict=None, cmap='Greys'):
    """
    Build a styled flat table for [cof_name][comx][metric] stats.
    """
    df = build_full_stats_table(
        ensemb_vsref_stats=ensemb_vsref_stats,
        xpx_dict=xpx_dict,
        metric=metric
    )

    if format_dict is None:
        format_dict = {
            'mean': '{:.2f}',
            'median': '{:.2f}',
            'low': '{:.2f}',
            'high': '{:.2f}'
        }

    def highlight_significant(data):
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        for idx in data.index:
            if data.loc[idx, 'signif']:
                styles.loc[idx, :] = 'font-weight: bold'
        return styles

    styled = df.style \
        .format(format_dict) \
        .apply(highlight_significant, axis=None) \
        .background_gradient(subset=['mean', 'median'], axis=0, cmap=cmap)

    return styled, df


def save_styled_results_table(
    styled_scores,
    results_scores,
    significance_scores,
    output_base,
    metric,
    reference,
    table_output_base=None,
):
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    table_output_base = Path(table_output_base or output_base)
    table_output_base.parent.mkdir(parents=True, exist_ok=True)

    metric_labels = {
        'mses': 'MSE',
        'rmses': 'RMSE',
        'corrs': 'correlation',
        'epochs': 'training epochs',
    }
    metric_label = metric_labels.get(metric, metric)
    caption = f"Median bootstrap {metric_label} change vs {reference} (%)"
    styled_scores.set_caption(caption)
    styled_scores.to_html(table_output_base.with_suffix('.html'))
    styled_scores.to_excel(
        table_output_base.with_name(table_output_base.name + '_styled.xlsx'),
        sheet_name='Results',
    )

    values = results_scores.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if len(finite_values):
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
    else:
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    color_map = plt.get_cmap('Greys')
    normalization = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    cell_colours = [
        [
            color_map(normalization(value)) if np.isfinite(value) else 'white'
            for value in row
        ]
        for row in values
    ]
    figure_width = max(8.0, 2.0 * (len(results_scores.columns) + 1))
    figure_height = max(3.0, 0.55 * (len(results_scores.index) + 1) + 1.2)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    axis.axis('off')
    table = axis.table(
        cellText=[
            [f'{value:.2f}' if np.isfinite(value) else '' for value in row]
            for row in values
        ],
        rowLabels=[str(label) for label in results_scores.index],
        colLabels=[str(label) for label in results_scores.columns],
        cellColours=cell_colours,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(-1, len(results_scores.columns))))
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#b7b7b7')
        if row == 0:
            cell.set_facecolor('#e6e6e6')
            cell.set_text_props(fontweight='bold')
        elif column == -1:
            cell.set_facecolor('#f2f2f2')
            cell.set_text_props(fontweight='bold')
        elif 0 <= row - 1 < len(significance_scores.index):
            cell.set_text_props(
                fontweight=(
                    'bold'
                    if significance_scores.iloc[row - 1, column]
                    else 'normal'
                )
            )
    table.scale(1.0, 1.35)
    axis.set_title(
        f'{caption}\n'
        'Bold: 5th–95th bootstrap interval excludes zero',
        fontsize=11,
        pad=18,
    )
    figure.savefig(
        output_base.with_suffix('.png'),
        dpi=200,
        bbox_inches='tight',
    )
    plt.close(figure)




























#=======================================

def compute_pairwise_comparison(ensemb_dict, cof_name1, comx1, cof_name2, comx2, 
                                pf_args=None):
    """
    Compare two specific (cof_name, comx) combinations from ensemb_dict.
    
    Args:
        ensemb_dict: ensemble dictionary from compute_ensemble_dict()
        cof_name1: first configuration name
        comx1: first combination identifier
        cof_name2: second configuration name (reference)
        comx2: second combination identifier (reference)
        pf_args: arguments for PairwiseFuncer (default: {'mode1': [se.RelativeChange, {'perc': True}]})
    
    Returns:
        dict with 'scores' and 'epochs' arrays
    """
    if pf_args is None:
        pf_args = {'mode1': [se.RelativeChange, {'perc': True}]}
    
    # Get data from first combination
    mses1, rmses1, corrs1, epochs1  = [ensemb_dict[cof_name1][comx1][x] 
                                       for x in ['mses', 'rmses', 'corrs', 'epochs']]
    # Get data from second combination (reference)
    mses2, rmses2, corrs2, epochs2  = [ensemb_dict[cof_name2][comx2][x] 
                                       for x in ['mses', 'rmses', 'corrs', 'epochs']]
    
    # Compute comparisons
    result = {
        'mses': np.array(se.PairwiseFuncer(mses1, mses2, **pf_args)),
        'rmses': np.array(se.PairwiseFuncer(rmses1, rmses2, **pf_args)),
        'corrs': np.array(se.PairwiseFuncer(corrs1, corrs2, **pf_args)),
        'epochs': np.array(se.PairwiseFuncer(epochs1, epochs2, **pf_args))
    }
    
    return result


def compute_pairwise_comparison_stats(ensemb_dict, cof_name1, comx1, cof_name2, comx2,
                                      percentile_low=2.5, percentile_high=97.5, 
                                      pf_args=None):
    """
    Compute statistics for a pairwise comparison between two specific combinations.
    
    Args:
        ensemb_dict: ensemble dictionary from compute_ensemble_dict()
        cof_name1: first configuration name
        comx1: first combination identifier
        cof_name2: second configuration name (reference)
        comx2: second combination identifier (reference)
        percentile_low: lower percentile for confidence interval (default: 2.5)
        percentile_high: upper percentile for confidence interval (default: 97.5)
        pf_args: arguments for PairwiseFuncer (default: {'mode1': [se.RelativeChange, {'perc': True}]})
    
    Returns:
        dict with nested structure: {metric: {stat_name: value}}
        Same structure as compute_stats_for_condition output
    """
    # Get comparison arrays
    comparison = compute_pairwise_comparison(ensemb_dict, cof_name1, comx1, 
                                             cof_name2, comx2, pf_args)
    
    results = {}
    
    for metric in ['mses', 'rmses', 'corrs', 'epochs']:
        data = comparison[metric]
        
        mean = np.mean(data)
        median = np.median(data)
        low_interval = np.percentile(data, percentile_low)
        high_interval = np.percentile(data, percentile_high)
        signif = 0 < low_interval or 0 > high_interval
        
        results[metric] = {
            'mean': mean,
            'median': median,
            'low': low_interval,
            'high': high_interval,
            'signif': signif
        }
    
    return results







#=======================================

# PIPELINES

def ResultsBasic(xpx_dict, tri_tpacks, bin_sizes, split, 
                 metric = 'mses', 
                 undersamp_props = None, ref_idx = 0, rewrite = False,
                 styled_output_base = None, table_output_base = None):
    
    ref_comx = xpx_dict['combs'][ref_idx][1][0]

    #----------------------- 
    pn_ensemb_dict = xpx_dict['pn'] + 'ensemb_dict' + '.p'
    if not os.path.exists(pn_ensemb_dict) or rewrite:
        ensemb_dict = compute_ensemble_dict(xpx_dict=xpx_dict,
                                            tri_tpacks = tri_tpacks, bin_sizes = bin_sizes, 
                                            split = split)
    
    #----------------------- 
    pn_ensemb_vsref = xpx_dict['pn'] + f'ensemb_vsref_{ref_comx}' + '.p'
    if not os.path.exists(pn_ensemb_vsref) or rewrite:
        ensemb_dict = se.PickleLoad(pn_ensemb_dict)
        ensemb_vsref = compute_vsref_dict(xpx_dict = xpx_dict, 
                                        ensemb_dict=ensemb_dict, ref_idx=ref_idx, 
                                        undersamp_props=undersamp_props)
    
    #-----------------------
    pn_ensemb_vsref_stats = xpx_dict['pn'] + f'ensemb_vsref_{ref_comx}_stats' + '.p'
    if not os.path.exists(pn_ensemb_vsref_stats) or rewrite:
        ensemb_vsref = se.PickleLoad(pn_ensemb_vsref)
        ensemb_vsref_stats = compute_vsref_stats_dict(
            xpx_dict = xpx_dict,
            ensemb_vsref=ensemb_vsref,
            ref_idx=ref_idx,
            percentile_low=5,
            percentile_high=95)
    

    ensemb_vsref_stats = se.PickleLoad(pn_ensemb_vsref_stats)

    styled_scores, results_scores, sig_scores = build_vsref_stats_table(ensemb_vsref_stats, xpx_dict, 
                                                                        metric=metric, use_median=True)
    
    df_stats = build_full_stats_table(ensemb_vsref_stats, xpx_dict, metric=metric)
    if styled_output_base is not None:
        reference = xpx_dict['combs'][ref_idx][1][0]
        save_styled_results_table(
            styled_scores,
            results_scores,
            sig_scores,
            styled_output_base,
            metric,
            reference,
            table_output_base,
        )

    return styled_scores, results_scores, df_stats

    