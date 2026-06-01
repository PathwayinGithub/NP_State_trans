import os
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from scipy.stats import wilcoxon as scipy_wilcoxon

from analysis_core import TRIAL_TYPES, CONDITIONS, get_expected_state

# =====================================================================
#  Constants
# =====================================================================

_FAMILIES = {
    'transition':  ['ablate_n2w', 'ablate_w2n'],
}
_ALL_ABLATION = [c for fam in _FAMILIES.values() for c in fam]

_TT_SHORT = {
    'nrem_to_wake': 'N→W', 'wake_to_nrem': 'W→N',
    'nrem_only': 'N Only', 'wake_only': 'W Only',
}
_COND_SHORT = {
    'ablate_n2w': 'Cut N→W', 'ablate_w2n': 'Cut W→N'
}


# =====================================================================
#  Formatting helpers
# =====================================================================

def format_pvalue(p, style='scientific'):
    if style in ('stars', 'both'):
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'
        if style == 'stars':
            return stars

    if p >= 0.05:
        sci = f"{p:.3f}"
    elif p >= 0.001:
        sci = f"{p:.4f}"
    elif p == 0.0 or p < 1e-300:
        sci = "< 1e-300"
    else:
        sci = f"{p:.2e}"

    if style == 'both':
        return f"{stars} ({sci})"
    return sci


def _significance_label(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'n.s.'


def _sanitize_for_json(obj):
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# =====================================================================
#  Holm-Bonferroni correction
# =====================================================================

def _holm_bonferroni_correct(p_dict):
    items = sorted(p_dict.items(), key=lambda x: x[1])
    k = len(items)
    corrected = {}
    running_max = 0.0
    for i, (key, p) in enumerate(items):
        adj = min(p * (k - i), 1.0)
        adj = max(adj, running_max)
        corrected[key] = adj
        running_max = adj
    return corrected


# =====================================================================
#  Core: Wilcoxon signed-rank (one-tailed)
# =====================================================================

def _paired_wilcoxon(group1, group2, alternative='greater'):
    """Wilcoxon signed-rank test (one-tailed).

    Parameters
    ----------
    group1, group2 : array-like
        Paired observations (same length).
    alternative : str
        'greater' — H1: group1 > group2
        'less'    — H1: group1 < group2

    Returns
    -------
    dict with keys compatible with downstream report / vis code.
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)
    n = len(group1)
    diff = group1 - group2
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

    result = {
        'n': int(n),
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'mean_group1': float(np.mean(group1)),
        'mean_group2': float(np.mean(group2)),
        'std_group1': float(np.std(group1, ddof=1)) if n > 1 else 0.0,
        'std_group2': float(np.std(group2, ddof=1)) if n > 1 else 0.0,
        'cohens_d': float(mean_diff / std_diff) if std_diff > 1e-12
                    else 0.0,
        'test': 'wilcoxon_signed_rank',
        'alternative': alternative,
    }

    if n < 2:
        result.update({'statistic': 0.0, 'p_value': 1.0})
        return result

    # Remove zero differences (standard Wilcoxon procedure)
    nonzero = diff[np.abs(diff) > 1e-12]
    result['n_nonzero'] = int(len(nonzero))

    if len(nonzero) < 1:
        result.update({'statistic': 0.0, 'p_value': 1.0})
        return result

    try:
        stat, pval = scipy_wilcoxon(nonzero, alternative=alternative)
        result.update({'statistic': float(stat), 'p_value': float(pval)})
    except Exception:
        result.update({'statistic': 0.0, 'p_value': 1.0,
                       'test': 'wilcoxon_failed'})

    return result


def _one_sample_above_chance(values, chance=1.0 / 2):
    """Wilcoxon signed-rank: H1: median of values > chance."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1)) if n > 1 else 0.0

    result = {
        'test': 'wilcoxon_one_sample',
        'n': int(n),
        'mean': m,
        'std': s,
        'chance': chance,
    }

    if n < 2:
        result.update({'statistic': 0.0, 'p_value': 1.0})
        return result

    shifted = values - chance
    nonzero = shifted[np.abs(shifted) > 1e-12]

    if len(nonzero) < 1:
        result.update({'statistic': 0.0, 'p_value': 1.0})
        return result

    try:
        stat, pval = scipy_wilcoxon(nonzero, alternative='greater')
        result.update({'statistic': float(stat), 'p_value': float(pval)})
    except Exception:
        result.update({'statistic': 0.0, 'p_value': 1.0})

    return result


# =====================================================================
#  Paired ablation tests (Holm per family)
# =====================================================================

def _run_paired_ablation_tests(per_model_metric, metric_name='accuracy'):
    """Paired Wilcoxon tests: each ablation condition vs intact.

    per_model_metric : {trial_type: {condition: [values_per_seed]}}
    metric_name      : 'accuracy'  → H1: intact > ablated  (decrease)
    """
    alternative = 'greater'

    results = {}
    for trial_type in TRIAL_TYPES:
        results[trial_type] = {}
        intact_vals = per_model_metric.get(trial_type, {}).get(
            'intact', [])
        if len(intact_vals) < 2:
            continue
        for condition in _ALL_ABLATION:
            abl_vals = per_model_metric.get(trial_type, {}).get(
                condition, [])
            if len(abl_vals) != len(intact_vals):
                continue
            results[trial_type][condition] = _paired_wilcoxon(
                intact_vals, abl_vals, alternative=alternative)

        # Holm-Bonferroni within each family
        for family_name, family_conditions in _FAMILIES.items():
            raw_ps = {c: results[trial_type][c]['p_value']
                      for c in family_conditions
                      if c in results[trial_type]}
            if not raw_ps:
                continue
            corrected = _holm_bonferroni_correct(raw_ps)
            for c, pc in corrected.items():
                results[trial_type][c]['p_corrected'] = pc
                results[trial_type][c]['family'] = family_name

                sig = _significance_label(pc)
                r = results[trial_type][c]
                d_str = (f"{r['cohens_d']:.2f}"
                         if abs(r.get('cohens_d', 0)) < 100 else '∞')
                print(f"     {_TT_SHORT[trial_type]:>6} × "
                      f"{_COND_SHORT.get(c, c):<18} "
                      f"Δ={r['mean_diff']:+.1%}  d={d_str}  "
                      f"p_holm={pc:.4f} {sig}  [{r['test']}]")
    return results


def _run_double_dissociation(per_model_accuracy, n_models):
    """Test: specific pathway drop > cross pathway drop."""
    dd = {}

    # N→W trial: Cut N→W (specific) vs Cut W→N (cross)
    drop_spec = [
        per_model_accuracy['nrem_to_wake']['intact'][i]
        - per_model_accuracy['nrem_to_wake']['ablate_n2w'][i]
        for i in range(n_models)]
    drop_cross = [
        per_model_accuracy['nrem_to_wake']['intact'][i]
        - per_model_accuracy['nrem_to_wake']['ablate_w2n'][i]
        for i in range(n_models)]
    dd['n2w'] = _paired_wilcoxon(drop_spec, drop_cross,
                                  alternative='greater')
    sig = _significance_label(dd['n2w']['p_value'])
    print(f"     N→W: spec_drop={np.mean(drop_spec):.1%} vs "
          f"cross_drop={np.mean(drop_cross):.1%}  "
          f"p={dd['n2w']['p_value']:.4f} {sig}")

    # W→N trial: Cut W→N (specific) vs Cut N→W (cross)
    drop_spec = [
        per_model_accuracy['wake_to_nrem']['intact'][i]
        - per_model_accuracy['wake_to_nrem']['ablate_w2n'][i]
        for i in range(n_models)]
    drop_cross = [
        per_model_accuracy['wake_to_nrem']['intact'][i]
        - per_model_accuracy['wake_to_nrem']['ablate_n2w'][i]
        for i in range(n_models)]
    dd['w2n'] = _paired_wilcoxon(drop_spec, drop_cross,
                                  alternative='greater')
    sig = _significance_label(dd['w2n']['p_value'])
    print(f"     W→N: spec_drop={np.mean(drop_spec):.1%} vs "
          f"cross_drop={np.mean(drop_cross):.1%}  "
          f"p={dd['w2n']['p_value']:.4f} {sig}")

    return dd


# =====================================================================
#  Main entry: statistics for one temporal window
# =====================================================================

def compute_statistics_for_window(temporal_result, window_name):
    """Full statistical tests for one ablation window (2-class).

    Reads per_model_accuracy from TemporalAblationResult.

    Returns
    -------
    stats : dict  (JSON-safe, same structure expected by vis code)
    """
    per_model_accuracy = temporal_result.per_model_accuracy.get(window_name)
    if per_model_accuracy is None:
        print(f"  No data for window '{window_name}'")
        return None

    n_models = temporal_result.n_models
    if n_models < 2:
        print("  <2 models: cannot run paired tests")
        return None

    window_label = temporal_result.window_labels.get(window_name,
                                                     window_name)
    print(f"\n{'=' * 60}")
    print(f"  Statistics — {window_label}  (n={n_models} models, "
          f"Wilcoxon signed-rank, Holm-corrected)")
    print(f"{'=' * 60}")

    stats = {
        'meta': {
            'n_models': n_models,
            'window': window_name,
            'window_label': window_label,
            'test': 'wilcoxon_signed_rank_one_tailed',
            'correction': 'holm_bonferroni',
            'families': _FAMILIES,
            'alpha': 0.05,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        },
        'tests': {},
        'significance_markers': {'accuracy': {}},
    }

    # 1. Intact above chance (H1: accuracy > 1/2)
    print(f"\n  1. Intact above chance (H1: median > 50%):")
    intact_tests = {}
    for trial_type in TRIAL_TYPES:
        intact_vals = per_model_accuracy.get(trial_type, {}).get(
            'intact', [])
        r = _one_sample_above_chance(intact_vals, chance=1.0 / 2)
        intact_tests[trial_type] = r
        sig = _significance_label(r['p_value'])
        print(f"     {_TT_SHORT[trial_type]:<8} mean={r['mean']:.1%}  "
              f"p={format_pvalue(r['p_value'])} {sig}")
    stats['tests']['intact_above_chance'] = intact_tests

    # 2. Ablation → accuracy decrease (H1: intact > ablated)
    print(f"\n  2. Ablation → accuracy (H1: intact > ablated):")
    abl_accuracy = _run_paired_ablation_tests(per_model_accuracy,
                                              'accuracy')
    stats['tests']['ablation_accuracy'] = abl_accuracy

    # 3. Double dissociation (H1: spec_drop > cross_drop)
    print(f"\n  3. Double dissociation (H1: spec_drop > cross_drop):")
    try:
        dd = _run_double_dissociation(per_model_accuracy, n_models)
    except (KeyError, IndexError) as e:
        print(f"     Skipped: {e}")
        dd = {}
    stats['tests']['double_dissociation'] = dd

    # 4. Significance markers (for vis code)
    for trial_type in TRIAL_TYPES:
        stats['significance_markers']['accuracy'][trial_type] = {}
        for condition in _ALL_ABLATION:
            if condition in abl_accuracy.get(trial_type, {}):
                p = abl_accuracy[trial_type][condition].get(
                    'p_corrected',
                    abl_accuracy[trial_type][condition]['p_value'])
                stats['significance_markers']['accuracy'][
                    trial_type][condition] = _significance_label(p)

    stats = _sanitize_for_json(stats)
    return stats


# =====================================================================
#  Text report
# =====================================================================

def generate_statistics_text_report(stats, save_path=None):
    """Human-readable text report from statistics dict."""
    meta = stats.get('meta', {})
    n = meta.get('n_models', '?')
    corr = meta.get('correction', 'holm_bonferroni')
    test_method = meta.get('test', 'wilcoxon_signed_rank_one_tailed')
    wlabel = meta.get('window_label', '')
    W = 95

    lines = [
        '=' * W,
        f'  ABLATION STATISTICAL TESTS  '
        f'(n = {n} models, window = {wlabel})',
        f'  Test: {test_method}  |  Correction: {corr}',
        '=' * W,
    ]

    # Intact above chance
    intact = stats.get('tests', {}).get('intact_above_chance', {})
    if intact:
        lines.append(
            '\n  Intact above chance (Wilcoxon, H1: median > 50%)')
        lines.append('  ' + '-' * (W - 2))
        for trial_type in TRIAL_TYPES:
            r = intact.get(trial_type, {})
            m = r.get('mean', 0)
            p = r.get('p_value', 1)
            sig = _significance_label(p)
            lines.append(f'    {_TT_SHORT.get(trial_type, trial_type):<8} '
                         f'mean = {m:.1%}   '
                         f'p = {format_pvalue(p)}  {sig}')

    # Accuracy table
    abl_acc = stats.get('tests', {}).get('ablation_accuracy', {})
    if abl_acc:
        lines.append('\n  Accuracy decrease: ablated vs intact '
                     '(Wilcoxon one-tailed, Holm-corrected)')
        lines.append('  ' + '-' * (W - 2))
        lines.append(f'    {"Trial":<8} {"Condition":<18} '
                     f'{"Intact":>8} {"Ablated":>8} {"Δ":>8} '
                     f'{"d":>7} {"p_corr":>10} {"Sig":>5}')
        lines.append('    ' + '-' * (W - 4))
        for trial_type in TRIAL_TYPES:
            for condition in _ALL_ABLATION:
                r = abl_acc.get(trial_type, {}).get(condition)
                if r is None:
                    continue
                iv = (r.get('mean_group1') or 0) * 100
                av = (r.get('mean_group2') or 0) * 100
                delta = iv - av
                d = r.get('cohens_d', 0)
                pc = r.get('p_corrected', r.get('p_value', 1))
                sig = _significance_label(pc)
                d_str = f'{d:+.2f}' if abs(d) < 100 else '    ∞'
                lines.append(
                    f'    {_TT_SHORT.get(trial_type, trial_type):<8} '
                    f'{_COND_SHORT.get(condition, condition):<18} '
                    f'{iv:>7.1f}% {av:>7.1f}% {delta:>+7.1f}% '
                    f'{d_str:>7} {format_pvalue(pc):>10} {sig:>5}')

    # Double dissociation
    dd = stats.get('tests', {}).get('double_dissociation', {})
    if dd:
        lines.append('\n  Double dissociation '
                     '(H1: specific drop > cross drop)')
        lines.append('  ' + '-' * (W - 2))
        for key, label in [('n2w', 'N→W'), ('w2n', 'W→N')]:
            r = dd.get(key, {})
            if r:
                md = r.get('mean_diff', 0)
                p = r.get('p_value', 1)
                sig = _significance_label(p)
                lines.append(
                    f'    {label}: specific−cross gap = {md:+.1%}   '
                    f'p = {format_pvalue(p)}  {sig}')

    lines.append('\n' + '=' * W)
    report = '\n'.join(lines)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  Report saved: {save_path}")

    return report