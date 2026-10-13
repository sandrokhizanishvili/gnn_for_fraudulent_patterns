'''Main results chart for the progress-meeting slides, built from Outputs/<FAMILY>/batch_summary.csv.

Run from the repo root or from Presentation/:  python Presentation/make_figures.py
Writes Presentation/figures/fig_results.pdf, fig_daily.pdf, fig_rwpe.pdf, fig_tuned.pdf, sized 1:1 to the slide text width (5.5 in) so
progress_meeting_sandrokh_131026.tex includes it without rescaling. Every number comes from the four CSV files.
'''
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE    = os.path.dirname(os.path.abspath(__file__))
REPO    = os.path.dirname(HERE)
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SLIDE_TEXT_WIDTH_IN = 5.5     # beamer 16:9 text width (160 mm paper, 1 cm margins)

# the four operators, in slide order (folder name, label)
FAMILIES = [('GIN', 'GIN'), ('GAT', 'GATv2'), ('PNA', 'PNA'), ('TRANSFORMER', 'Transformer')]
# the seven feature configurations in the order of EXPERIMENTS.md (mp / readout)
CONFIGS = [('none', 'base'), ('base', 'base'), ('none', 'full'), ('full', 'full'),
           ('base', 'full'), ('base', 'gfp'), ('none', 'gfp')]
CONFIG_LABELS = ['1  MP: none  ·  Readout: base',
                 '2  MP: base  ·  Readout: base',
                 '3  MP: none  ·  Readout: base + GFP',
                 '4  MP: base + GFP  ·  Readout: base + GFP',
                 '5  MP: base  ·  Readout: base + GFP',
                 '6  MP: base  ·  Readout: GFP only',
                 '7  MP: none  ·  Readout: GFP only']
# one colour per configuration, the same in every operator group (Okabe-Ito, colour-blind safe)
CONFIG_COLOURS = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
NOTE = ('MP = edge features inside message passing;  Readout = edge features given to the final classifier;\n'
        'base = 20 transaction features;  GFP = 61 graph-pattern features;  '
        'diamond = highest test F1 of the operator.')

# ---- one shared academic style for every figure ----
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 9,
    'legend.fontsize': 7.5, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': 'black', 'axes.linewidth': 0.8,
    'xtick.color': 'black', 'ytick.color': 'black', 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': '#dddddd', 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'pdf.fonttype': 42,
})


def load_summaries():
    '''Return {operator label: DataFrame indexed by config number 1..7} from the four batch_summary.csv files.'''
    out = {}
    for folder, label in FAMILIES:
        df = pd.read_csv(os.path.join(REPO, 'Outputs', folder, 'batch_summary.csv'))
        rows = []
        for mp, readout in CONFIGS:
            match = df[(df['mp'] == mp) & (df['readout'] == readout)]
            assert len(match) == 1, f'{folder}: config {mp}/{readout} found {len(match)} times'
            rows.append(match.iloc[0])
        out[label] = pd.DataFrame(rows).reset_index(drop=True)
        out[label].index = range(1, 8)
    return out


def results_chart(data):
    '''Grouped bars: one group per operator, seven configurations per group, test F1; highest test F1 marked.'''
    fig, ax = plt.subplots(figsize=(SLIDE_TEXT_WIDTH_IN, 1.95))
    fig.subplots_adjust(left=0.11, right=0.53, bottom=0.27, top=0.96)
    width = 0.115
    xs = np.arange(len(FAMILIES))
    for cfg in range(1, 8):
        values = [100 * data[label].loc[cfg, 'test_f1'] for _, label in FAMILIES]
        ax.bar(xs + (cfg - 4) * width, values, width=width - 0.012, color=CONFIG_COLOURS[cfg - 1],
               label=CONFIG_LABELS[cfg - 1], zorder=2)
    for x, (_, label) in zip(xs, FAMILIES):
        best = data[label]['test_f1'].idxmax()
        ax.plot(x + (best - 4) * width, 100 * data[label].loc[best, 'test_f1'] + 4, marker='D',
                markersize=2.8, color='black', linestyle='none', zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label in FAMILIES])
    ax.set_xlim(-0.5, len(FAMILIES) - 0.5)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel('F1 (%)')
    ax.tick_params(axis='x', length=0)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.06), handlelength=1.1, handleheight=0.9,
              labelspacing=0.45, handletextpad=0.5, borderaxespad=0)
    fig.text(0.11, 0.015, NOTE, fontsize=6.5, color='#444444', va='bottom', ha='left', linespacing=1.4)
    fig.savefig(os.path.join(FIG_DIR, 'fig_results.pdf'))
    plt.close(fig)


VARIANTS = {   # folder under Outputs/, summary file name, legend label
    'rwpe':  ('RWPE',      'batch_summary_rwpe.csv',      'RWPE'),
    'tuned': ('GFP_TUNED', 'batch_summary_gfp_tuned.csv', 'GFP V1'),   # the sheet the repo calls `tuned'
}
PAIR_CONFIGS = [(5, 'base', 'full', '#0072B2'), (4, 'full', 'full', '#F0E442')]   # config, mp, readout, colour (as in fig_results)
EDGE = {'#0072B2': '#0072B2', '#F0E442': '#9A8F00'}   # marker outline: the yellow needs a darker edge on white
TIE = 0.02


def load_variant(key):
    '''Return {operator label: DataFrame indexed by config (5, 4)} for one variant batch.'''
    folder, csv_name, _ = VARIANTS[key]
    out = {}
    for fam_folder, label in FAMILIES:
        df = pd.read_csv(os.path.join(REPO, 'Outputs', folder, fam_folder, csv_name))
        rows = []
        for cfg, mp, readout, _ in PAIR_CONFIGS:
            match = df[(df['mp'] == mp) & (df['readout'] == readout)]
            assert len(match) == 1, f'{key} {fam_folder}: config {cfg} found {len(match)} times'
            rows.append(match.iloc[0])
        out[label] = pd.DataFrame(rows).reset_index(drop=True)
        out[label].index = [c for c, *_ in PAIR_CONFIGS]
    return out


SHADES = {   # per configuration colour (as in fig_results): V0, +RWPE (darker), +tuned GFP (lighter)
    '#0072B2': {'v0': '#0072B2', 'rwpe': '#004A75', 'tuned': '#8CC4E8'},
    '#F0E442': {'v0': '#F0E442', 'rwpe': '#B5A800', 'tuned': '#F9F3A8'},
}


def paired_chart(base, variant, key):
    '''Two panels (test F1, test PR-AUC). Per operator: config 5 as V0 and +variant side by side,
    a gap, then config 4 the same way. Colours per configuration as in fig_results; the variant bar
    is a darker (RWPE) or lighter (tuned GFP) shade of the same colour.'''
    _, _, name = VARIANTS[key]
    fig, axes = plt.subplots(1, 2, figsize=(SLIDE_TEXT_WIDTH_IN, 1.92))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.36, top=0.95, wspace=0.3)
    xs = np.arange(len(FAMILIES))
    width = 0.17
    offsets = {(5, 'v0'): -0.31, (5, key): -0.13, (4, 'v0'): 0.13, (4, key): 0.31}
    for ax, metric, ylabel in zip(axes, ('test_f1', 'test_pr_auc'), ('F1 (%)', 'PR-AUC (%)')):
        for x, (_, label) in zip(xs, FAMILIES):
            for cfg, _, _, colour in PAIR_CONFIGS:
                ax.bar(x + offsets[(cfg, 'v0')], 100 * base[label].loc[cfg, metric], width=width,
                       color=SHADES[colour]['v0'], edgecolor='#555555', linewidth=0.4, zorder=2)
                ax.bar(x + offsets[(cfg, key)], 100 * variant[label].loc[cfg, metric], width=width,
                       color=SHADES[colour][key], edgecolor='#555555', linewidth=0.4, zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([label for _, label in FAMILIES], fontsize=8.5)
        ax.set_xlim(-0.55, len(FAMILIES) - 0.45)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', labelsize=8)
    handles = []
    for (cfg, _, _, colour), cfg_name in zip(PAIR_CONFIGS, ('config 5 (base / base+GFP)', 'config 4 (base+GFP / base+GFP)')):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=SHADES[colour]['v0'], ec='#555555', lw=0.4, label=cfg_name + ': V0'))
        handles.append(plt.Rectangle((0, 0), 1, 1, color=SHADES[colour][key], ec='#555555', lw=0.4,
                                     label=cfg_name + (': V1' if key == 'tuned' else ': + ' + name)))
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=1.5, labelspacing=0.3)
    fig.savefig(os.path.join(FIG_DIR, f'fig_{key}.pdf'))
    plt.close(fig)


def print_variant_numbers(base, variants):
    '''Print validation gap, test F1 and test PR-AUC for every (operator, config): baseline, +RWPE, +tuned.'''
    print('\nvariant runs: operator config | val dF1 rwpe, tuned | test F1 base / +rwpe / +tuned | test PR-AUC base / +rwpe / +tuned')
    for _, label in FAMILIES:
        for cfg, *_ in PAIR_CONFIGS:
            b = base[label].loc[cfg]
            r = variants['rwpe'][label].loc[cfg]
            t = variants['tuned'][label].loc[cfg]
            print(f'  {label:<12} {cfg} | {r["val_f1"] - b["val_f1"]:+.3f}, {t["val_f1"] - b["val_f1"]:+.3f} | '
                  f'{b["test_f1"]:.3f} / {r["test_f1"]:.3f} / {t["test_f1"]:.3f} | '
                  f'{b["test_pr_auc"]:.3f} / {r["test_pr_auc"]:.3f} / {t["test_pr_auc"]:.3f}')


def daily_counts():
    '''Transactions and laundering transactions per day of the raw HI-Small file (cached as a small CSV).'''
    cache = os.path.join(FIG_DIR, 'daily_counts.csv')
    if os.path.exists(cache):
        return pd.read_csv(cache, parse_dates=['day'])
    raw = pd.read_csv(os.path.join(REPO, 'Data', 'HI-Small_Trans.csv'), usecols=['Timestamp', 'Is Laundering'])
    day = pd.to_datetime(raw['Timestamp']).dt.normalize().rename('day')
    daily = raw.groupby(day)['Is Laundering'].agg(total='count', laundering='sum').reset_index()
    daily.to_csv(cache, index=False)
    return daily


def daily_chart(daily):
    '''Two stacked panels (shared x): all transactions per day, laundering per day; dashed line = truncation.'''
    cut = pd.Timestamp('2022-09-11')          # everything from Sep 11 on is dropped
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(2.5, 2.0), sharex=True)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.96, bottom=0.17, hspace=0.3)
    top.fill_between(daily['day'], daily['total'] / 1e6, color='#9ECAE1', linewidth=0, zorder=2)
    top.plot(daily['day'], daily['total'] / 1e6, color='#0072B2', linewidth=1.2, zorder=3)
    top.set_ylabel('All, millions / day', fontsize=7.5)
    top.set_ylim(0, None)
    bottom.plot(daily['day'], daily['laundering'], color='#D55E00', linewidth=1.2, zorder=3)
    bottom.set_ylabel('Laundering / day', fontsize=7.5)
    bottom.set_ylim(0, None)
    for ax in (top, bottom):
        ax.axvline(cut, color='black', linestyle='--', linewidth=0.8, zorder=4)
        ax.tick_params(labelsize=7)
    top.text(cut, top.get_ylim()[1] * 0.95, ' dropped ', fontsize=7, ha='left', va='top')
    bottom.set_xticks(pd.to_datetime(['2022-09-01', '2022-09-05', '2022-09-09', '2022-09-13', '2022-09-17']))
    bottom.set_xticklabels(['Sep 1', 'Sep 5', 'Sep 9', 'Sep 13', 'Sep 17'])
    bottom.set_xlim(pd.Timestamp('2022-08-31 12:00'), daily['day'].max() + pd.Timedelta(hours=12))
    fig.savefig(os.path.join(FIG_DIR, 'fig_daily.pdf'))
    plt.close(fig)


def print_table_numbers(data):
    '''Print the numbers used in the slide tables, so progress_meeting_sandrokh_131026.tex can be checked against the CSVs.'''
    print('test F1 (rows GIN, GATv2, PNA, Transformer; columns configs 1..7):')
    print(np.array([[data[label].loc[c, 'test_f1'] for c in range(1, 8)] for _, label in FAMILIES]).round(3))
    print('\nbest per operator by validation F1:')
    for _, label in FAMILIES:
        d = data[label]
        best = d['val_f1'].idxmax()
        ties = [c for c in range(1, 8) if d.loc[best, 'val_f1'] - d.loc[c, 'val_f1'] < 0.02]
        r = d.loc[best]
        print(f'  {label:<12} config {best} (ties {ties})  val F1 {r["val_f1"]:.3f}  test F1 {r["test_f1"]:.3f}  '
              f'PR-AUC {r["test_pr_auc"]:.3f}  params {int(r["params"]):,}')


if __name__ == '__main__':
    data = load_summaries()
    results_chart(data)
    daily_chart(daily_counts())
    variants = {key: load_variant(key) for key in VARIANTS}
    for key in VARIANTS:
        paired_chart(data, variants[key], key)
    print_table_numbers(data)
    print_variant_numbers(data, variants)
    print('\nfigures written to', FIG_DIR)
