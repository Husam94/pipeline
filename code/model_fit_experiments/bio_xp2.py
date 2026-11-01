''' THIS IS LOSS FUNCTION TESTING '''

import argparse
import numpy as np
import glob, shutil, os
import multiprocessing
#------------------------------------
import poseigen_seaside.basics as se
import poseigen_trident.utils as tu
import poseigen_trident.othermodels as othmods
import poseigen_trident.deepstarr as deepstarr
import poseigen_oyster.model as oyster_models
from poseigen_mimic import (
    Dual_Mimic,
    Mimic,
    Reset_Dual_Mimic,
    Reset_Mimic,
)
#------------------------------------
import bio_xps_functions as xpfus
#------------------------------------


def resolve_data_path():
    env_path = os.environ.get("BIO_XP_DATA_PATH")
    if env_path:
        return os.path.abspath(env_path)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-path")
    args, _ = parser.parse_known_args()

    if args.data_path:
        return os.path.abspath(args.data_path)

    raise RuntimeError("Missing data path. Pass --data-path or set BIO_XP_DATA_PATH.")

DATA_PATH = resolve_data_path()
os.chdir(DATA_PATH)

searchfolder = se.NewFolder('search')
xpsfolder = se.NewFolder('xps')

fit_args = se.PickleLoad(searchfolder + 'fit_args')
modx_configs = se.PickleLoad(xpsfolder + 'modx_configs')

xpx_dict = se.PickleLoad(xpsfolder + 'xp2_dict')
modx_names = xpx_dict['modx_names']
modx_names_full = xpx_dict['modx_names_full']

#------------------------------

tri_tpacks, bin_sizes, split_transaug = xpfus.load_fit_data(DATA_PATH)
if len(tri_tpacks) == 4: d_x, d_y_ms, d_s, d_b = tri_tpacks
else: d_x1, d_y_ms, d_s, d_b, d_x2 = tri_tpacks

d_s_rs = np.swapaxes(np.expand_dims(d_s, axis=-1), 1, -1)

print('LOADED')


def run_job(args):

    modx_name, modx_name_full, com = args
    cof = modx_configs[modx_name]

    argoz = se.PickleLoad(f'{searchfolder}{modx_name}/RO_args')
    RO_folo = argoz['pn']
    best_config_idx = se.PickleLoad(RO_folo + 'best_config_idx')

    pn_t1 = se.NewFolder(f"{xpx_dict['pn']}{modx_name_full}")
    pn_t5 = se.NewFolder(pn_t1 + str(com))

    #------------------------------------------

    lmo = fit_args['MSE_mode']
    mmo = fit_args['MSE_mode']
    
    #------------------------------------------

    los = com
    bw_mode = None 
    if los =='MSE-C': bw_mode = 'comp'
    elif los =='MSE-R': bw_mode = 'recip'

    #------------------------------------------

    if modx_name == 'Dual_Oyster':
        modo = oyster_models.Dual_Oyster
        mim = [oyster_models.Reset_Dual_Oyster, {}]
    
    elif modx_name.startswith('Dual_Mimic'):
        modo = Dual_Mimic
        mim = [Reset_Dual_Mimic, {}]
    
    elif modx_name == 'Oyster':
        modo = oyster_models.Oyster
        mim = [oyster_models.Reset_Oyster, {}]
    
    elif modx_name.startswith('Mimic'):
        modo = Mimic
        mim = [Reset_Mimic, {}]
    
    elif modx_name == 'DeepSTARR': 
        modo = deepstarr.DeepSTARR
        mim = [deepstarr.Reset_DeepSTARR, {}]
    
    elif modx_name == 'BiChrom': 
        modo  = othmods.BiChrom
        mim = [othmods.Reset_BiChrom, {}]

    #-----------------------------------------

    if com == 'MSE-R': 

        rep_folo = f'{RO_folo}Top_1.0/0_{best_config_idx}/'

        if not os.path.isdir(rep_folo):
            raise FileNotFoundError(f"Source folder does not exist: {rep_folo}")

        for name in os.listdir(rep_folo):
            src_path = os.path.join(rep_folo, name)
            dst_path = os.path.join(pn_t5, name)

            print(src_path, dst_path)

            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)
        
    #------------------------------------------
    
    xp2_TT_args = {'inps': [d_x] if len(tri_tpacks) == 4 else [d_x1, d_x2], 
                    'out': d_y_ms,
                    'out_std': None, 
                    'out_bind': d_b, 'bin_sizes': bin_sizes,    
                    'Split': split_transaug,
                    'loss_mode': lmo, 'metrics_mode': mmo, 

                    'EUS_prop': xpx_dict['EUS_prop'], 'EUS_mode': xpx_dict['EUS_mode'],
                    
                    'mod_init_mode': mim,
                    
                    **fit_args['TT_base']}

    xp2_TT_args.update({'bw_mode': bw_mode})     


    xp2_TCS_args = {'trainer_args': xp2_TT_args, **fit_args['TCS_base']}
    xp2_TCR_args = {**xp2_TCS_args, **fit_args['TCR_base'], 
                    'repeats': xpx_dict['TCR_repeats'], 'pathname': pn_t5}

    tu.TridentCanRepeater(modo, cof, data=None, **xp2_TCR_args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", help="Working data directory for pickle I/O")
    parser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--skip_fit', action='store_true', help='Skip fitting step if specified')
    parser.add_argument('--resource_comp', action='store_true', help='Run resource comparison if specified')
    args = parser.parse_args()

    if not args.skip_fit: 

        jobs = []

        for modx_name, modx_name_full in zip(modx_names, modx_names_full):
            
            for com in xpx_dict['loss_funcs']:

                #if com != 'MSE-R': continue
                
                jobs.append((modx_name, modx_name_full, com))
        
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=args.n_workers) as pool:
            pool.map(run_job, jobs)
        

    #-----------------------------------------------

    figures_folder = se.NewFolder(
        os.path.join(DATA_PATH, 'output', 'figures', 'model_fitting')
    )
    tables_folder = se.NewFolder(
        os.path.join(DATA_PATH, 'output', 'tables', 'model_fitting')
    )

    styled_scores, results_scores, df_stats = xpfus.ResultsBasic(
            xpx_dict,
            tri_tpacks,
            bin_sizes,
            split_transaug,
            undersamp_props=None,
            metric="rmses",
            ref_idx=0,
            rewrite=True,
            styled_output_base=figures_folder + 'xp2_rmses_results_scores',
            table_output_base=tables_folder + 'xp2_rmses_results_scores',
        )

    results_scores.to_excel(tables_folder + 'xp2_' + 'rmses_results_scores.xlsx')
    df_stats.to_excel(tables_folder + 'xp2_' + 'rmses_df_stats.xlsx')
        
