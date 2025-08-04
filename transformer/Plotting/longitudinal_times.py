# longitudinal_times_notmissing_RMSE.py
# ‑‑ plots the relative RMSE (not‑missing) vs. time for MDiiN, HiDiNet‑T, and (optionally) Elastic‑net

import argparse, os, sys
from pathlib import Path

import numpy as np
import torch
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl

# ──── Matplotlib cosmetics ────────────────────────────────────────────────────
mpl.rcParams['mathtext.fontset'] = 'cm'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cm  = plt.get_cmap('Set1')   #   markers / medians
cm2 = plt.get_cmap('Set2')   #   box‑fill colours

# ──── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser('Predict longitudinal all times')
parser.add_argument('--job_id',        type=int, required=True)
parser.add_argument('--epoch',         type=int, required=True)
parser.add_argument('--lamp_job_id',   type=int, required=True)
parser.add_argument('--lamp_epoch',    type=int, required=True)
parser.add_argument('--dataset',       choices=['elsa', 'sample'],
                    default='elsa', help='training dataset used')
parser.add_argument('--no_compare',    action='store_true',
                    help='omit Elastic‑net baseline')
args = parser.parse_args()

postfix = '_sample' if args.dataset == 'sample' else ''
dir     = os.path.dirname(os.path.realpath(__file__))

# ──── Constants ──────────────────────────────────────────────────────────────
N = 29                       # number of health variables
years_grid = [0,2,4,6,8,10,12,14,16,18]   # bins for Δ‑time from baseline
dt, length = 0.5, 50

# ──── Population stats (for baseline RMSE) ───────────────────────────────────
pop_avg     = np.load(f'{dir}/../../Data/Population_averages{postfix}.npy')
pop_avg_env = np.load(f'{dir}/../../Data/Population_averages_env{postfix}.npy')
pop_std     = np.load(f'{dir}/../../Data/Population_std{postfix}.npy')
pop_avg_    = torch.from_numpy(pop_avg[...,1:]).float()
pop_avg_env = torch.from_numpy(pop_avg_env).float()
pop_std     = torch.from_numpy(pop_std[...,1:]).float()
pop_avg_bins = np.arange(40,105,3)[:-2]

# ──── DataLoader setup ───────────────────────────────────────────────────────
file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))   # project root

from DataLoader.dataset   import Dataset
from DataLoader.collate   import custom_collate
from Utils.transformation import Transformation

test_csv   = f'{dir}/../../Data/test{postfix}.csv'
test_set   = Dataset(test_csv, N, pop=False, min_count=10)
num_test   = len(test_set)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=num_test, shuffle=False,
    collate_fn=lambda x: custom_collate(x, pop_avg_, pop_avg_env,
                                        pop_std, 1.0))

# ──── Transformation object ─────────────────────────────────────────────────
mean_def  = read_csv(f'{dir}/../../Data/mean_deficits{postfix}.txt',
                     index_col=0, header=None).values[1:].flatten()
std_def   = read_csv(f'{dir}/../../Data/std_deficits{postfix}.txt',
                     index_col=0, header=None).values[1:].flatten()
log_idx   = [6,7,15,16,23,25,26,28]      # log‑scaled vars
psi = Transformation(mean_def[:-3], std_def[:-3], log_idx)

# ──── Load model prediction matrices ────────────────────────────────────────
mean = np.load(
    dir + f'/../../Analysis_Data/Mean_trajectories_job_id{args.job_id}'
          f'_epoch{args.epoch}_DJIN{postfix}.npy')

# HiDiNet‑T (dataset‑aware path)
lamp = np.load(
    dir + f'/../Analysis_Data_{args.dataset}/Mean_trajectories_job_id'
          f'{args.lamp_job_id}_epoch{args.lamp_epoch}_LAMP{postfix}.npy')

if not args.no_compare:
    linear = np.load(
        dir + f'/../../Comparison_models/Predictions/Longitudinal_predictions_baseline_id1_rfmice{postfix}.npy')

# un‑transform model outputs
mean[:,:,1:]  = psi.untransform(mean[:,:,1:])
lamp[:,:,1:]  = psi.untransform(lamp[:,:,1:])
if not args.no_compare:
    linear[:,:,1:] = psi.untransform(linear[:,:,1:])
pop_avg_ = psi.untransform(pop_avg_.numpy())

# ──── storage arrays: [var][time_bin] lists ─────────────────────────────────
def make_lists():
    return [[[] for _ in range(10)] for _ in range(N)]

missing, notmissing     = make_lists(), make_lists()
lamp_notmissing         = make_lists()
if not args.no_compare:
    linear_notmissing   = make_lists()
exact_missing, exact_notmissing = make_lists(), make_lists()
weights_missing, weights_notmissing = make_lists(), make_lists()
pop_missing, pop_notmissing   = make_lists(), make_lists()
first_notmissing              = make_lists()

# ──── Walk through the big batch once per time bin ─────────────────────────
with torch.no_grad():
    for yi, years in enumerate(years_grid):

        for batch in test_loader: break
        y, times, mask = batch['Y'], batch['times'], batch['mask']
        sex_index = batch['env'][:,12].long().numpy()
        sample_wt = batch['weights'].numpy()

        y_np = psi.untransform(y.numpy())
        y_np = mask * y_np + (1-mask)*(-1000)

        # compact arrays per‑subject
        rec_times, rec_vals, rec_mask = [], [], []
        for b in range(num_test):
            obs = (mask[b].sum(dim=-1) > 0)
            rec_times.append(times[b,obs].numpy().astype(int))
            rec_vals.append( y_np[b,obs] )
            rec_mask.append( mask[b,obs].numpy().astype(int) )

        if yi == 0:          # we only fill bins 1..9 (Δ‑years > 0)
            continue

        for b in range(num_test):
            t_mdl = 0
            for t_r, t_val in enumerate(rec_times[b]):

                t_bin = np.digitize(t_val, pop_avg_bins, right=True)
                pop_vec = pop_avg_[sex_index[b], t_bin]

                while t_mdl < min(40, np.sum(~np.isnan(mean[b,:,1]))):
                    if t_val == mean[b, t_mdl, 0]:
                        for n in range(N):
                            if (rec_mask[b][t_r,n] > 0 and
                                 years-1 <= (t_val - rec_times[b][0]) < years+1):

                                if rec_mask[b][0,n] < 1:         # baseline missing
                                    missing[n][yi].append(mean[b,t_mdl,n+1])
                                    exact_missing[n][yi].append(rec_vals[b][t_r,n])
                                    weights_missing[n][yi].append(sample_wt[b])
                                    pop_missing[n][yi].append(pop_vec[n])
                                else:                            # baseline observed
                                    notmissing[n][yi].append(mean[b,t_mdl,n+1])
                                    lamp_notmissing[n][yi].append(lamp[b,t_mdl,n+1])
                                    exact_notmissing[n][yi].append(rec_vals[b][t_r,n])
                                    weights_notmissing[n][yi].append(sample_wt[b])
                                    first_notmissing[n][yi].append(rec_vals[b][0,n])
                                    pop_notmissing[n][yi].append(pop_vec[n])
                                    if not args.no_compare:
                                        linear_notmissing[n][yi].append(linear[b,t_mdl,n+1])
                        break
                    t_mdl += 1

# ──── Compute RMSE arrays ───────────────────────────────────────────────────
RMSE_not, RMSE_lamp, RMSE_pop, RMSE_first = (
    np.zeros((10,N)) for _ in range(4))
if not args.no_compare:
    RMSE_lin = np.zeros((10,N))

for yi in range(1,10):
    for n in range(N):
        w_n = np.array(weights_notmissing[n][yi])
        if len(w_n) < 25:                 # too little data
            RMSE_not[yi,n]   = np.nan
            RMSE_lamp[yi,n]  = np.nan
            RMSE_pop[yi,n]   = np.nan
            RMSE_first[yi,n] = np.nan
            if not args.no_compare:
                RMSE_lin[yi,n] = np.nan
            continue

        x_obs  = np.array(exact_notmissing[n][yi])
        x_mdl  = np.array(notmissing[n][yi])
        x_lamp = np.array(lamp_notmissing[n][yi])
        x_pop  = np.array(pop_notmissing[n][yi])
        x_1st  = np.array(first_notmissing[n][yi])

        RMSE_not[yi,n]   = np.sqrt((w_n*((x_obs-x_mdl )**2)).mean())
        RMSE_lamp[yi,n]  = np.sqrt((w_n*((x_obs-x_lamp)**2)).mean())
        RMSE_pop[yi,n]   = np.sqrt((w_n*((x_obs-x_pop )**2)).mean())
        RMSE_first[yi,n] = np.sqrt((w_n*((x_obs-x_1st)**2)).mean())
        if not args.no_compare:
            x_lin = np.array(linear_notmissing[n][yi])
            RMSE_lin[yi,n] = np.sqrt((w_n*((x_obs-x_lin)**2)).mean())

# normalise by population RMSE
Rel = RMSE_not / RMSE_pop
Rel_lamp = RMSE_lamp / RMSE_pop
Rel_first= RMSE_first / RMSE_pop
if not args.no_compare:
    Rel_lin  = RMSE_lin  / RMSE_pop

# ──── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4,5))

non_nurse = [0,3,4,8,9,10,11]
nurse     = [1,2,5,6,7,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]

for yi, yrs in enumerate(years_grid[:-2]):   # stop at 14‑year bin for plot
    if yrs == 0: continue
    
    # Boxplots for non-nurse and nurse data
    ax.boxplot([Rel[yi][non_nurse][~np.isnan(Rel[yi][non_nurse])]],
               positions=[yrs-0.25], widths=0.28, showfliers=False,
               patch_artist=True, boxprops=dict(facecolor=cm2(2),alpha=.6,color='k'),
               medianprops=dict(color=cm(4),lw=2))
    ax.boxplot([Rel[yi][nurse][~np.isnan(Rel[yi][nurse])]],
               positions=[yrs+0.25], widths=0.28, showfliers=False,
               patch_artist=True, boxprops=dict(facecolor=cm2(3),alpha=.6,color='k'),
               medianprops=dict(color=cm(4),lw=2))

    # median markers
    median_nn  = np.nanmedian(Rel[yi][non_nurse])
    median_lnn = np.nanmedian(Rel_lamp[yi][non_nurse])
    median_n   = np.nanmedian(Rel[yi][nurse])
    median_ln  = np.nanmedian(Rel_lamp[yi][nurse])  # HiDiNet-T median for nurse data
    
    # Plot MDiiN medians (circles)
    ax.plot(yrs-0.25, median_nn,  marker='o', color=cm(4), zorder=9)       # MDiiN non-nurse
    ax.plot(yrs+0.25, median_n,   marker='o', color=cm(4), zorder=9)       # MDiiN nurse
    
    # Plot HiDiNet-T medians (triangles)
    ax.plot(yrs-0.25, median_lnn, marker='v', color=cm(1), zorder=9)       # HiDiNet-T non-nurse
    ax.plot(yrs+0.25, median_ln,  marker='v', color=cm(1), zorder=9)       # HiDiNet-T nurse
    
    if not args.no_compare:
        median_lin_n  = np.nanmedian(Rel_lin[yi][nurse])
        median_lin_nn = np.nanmedian(Rel_lin[yi][non_nurse])
        ax.plot(yrs+0.25, median_lin_n,  marker='s', color=cm(2), zorder=8)
        ax.plot(yrs-0.25, median_lin_nn, marker='s', color=cm(2), zorder=8)

# axes & labels
ax.set_ylabel('Relative RMSE', fontsize=14)
ax.set_xlabel('Years from baseline', fontsize=14)
ax.set_ylim(0.6, 1.2)
ax.set_xlim(1, 15)
ax.set_xticks([2,4,6,8,10,12,14])
ax.set_xticklabels(['[1,2]','[3,4]','[5,6]','[7,8]',
                    '[9,10]','[11,12]','[13,14]'])
ax.yaxis.set_minor_locator(MultipleLocator(0.025))
ax.plot([1,15],[1,1],'k--',lw=1,label='Population mean')

# Legend
handles  = [
    plt.Line2D([0],[0],marker='s',lw=0,color='k',markerfacecolor=cm2(2),
               alpha=.6,label='2‑yr self‑report waves'),
    plt.Line2D([0],[0],marker='s',lw=0,color='k',markerfacecolor=cm2(3),
               alpha=.6,label='4‑yr nurse waves'),
    plt.Line2D([0],[0],marker='v',color=cm(1),lw=0,label='HiDiNet median'),
    plt.Line2D([0],[0],marker='o',color=cm(4),lw=0,label='RNN median'),
    
]
if not args.no_compare:
    handles.append(plt.Line2D([0],[0],marker='s',color=cm(2),lw=0,
                              label='Elastic‑net median'))
handles.append(plt.Line2D([0],[0],color='k',ls='--',lw=1,
                          label='Population mean'))
ax.legend(handles=handles, loc='upper left', fontsize=9)

ax.text(-0.05,1.05,'f',transform=ax.transAxes, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(dir + f'/../Plots/Longitudinal_times_notmissing_RMSE_job_id'
                 f'{args.lamp_job_id}_epoch{args.lamp_epoch}{postfix}.pdf')
print('Plot saved.')