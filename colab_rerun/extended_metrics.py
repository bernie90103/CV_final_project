"""Post-hoc extended metrics for completed PatchCore runs (Colab add-on).

Reads only each run's saved predictions.npz and samples.json; no model is
rebuilt. This file lives outside study/ so study_source_sha256 is unchanged.

Thresholds never use defect masks, with one labelled exception: pixel F1-max
picks its best threshold on the test masks and is reported only as an oracle
diagnostic. Region recall uses a threshold taken from normal test images only.
Subgroups follow study.metrics.evaluate_groups: every subgroup contains all
normal test images plus its own anomalous images.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_curve

from study.io import write_csv, write_json
from study.metrics import pro_curve, safe_auroc

GROUPS = ('all', 'small', 'large')
FPR_LIMITS = (.3, .05, .01)
IMAGE_FPR = .05
SIZE_EDGES = (0, 64, 256, 1024)
SIZE_LABELS = ('<64', '64-255', '256-1023', '>=1024')
LAYER_ORDER = ('layer2', 'layer3', 'fusion')
STRUCTURE = np.ones((3, 3), dtype=bool)


def metric_names(tolerance=8, normal_fpr=.01):
    return ([f'aupro@{limit:g}' for limit in FPR_LIMITS]
            + ['pixel_ap', 'pixel_f1max_oracle', 'image_auroc', 'image_ap', f'image_tpr@fpr{IMAGE_FPR:g}',
               'pointing@0px', f'pointing@{tolerance:g}px', f'region_recall@nfpr{normal_fpr:g}'])


def size_bin(n_pixels):
    """Size class of one connected defect region, in transformed-mask pixels."""
    if n_pixels < 1:
        raise ValueError('A region has at least one pixel.')
    return SIZE_LABELS[int(np.searchsorted(SIZE_EDGES, n_pixels, side='right')) - 1]


def aupro_from_curve(curve, limit):
    """Same clipping and integration as study.metrics.aupro, reusing one PRO curve."""
    if curve is None:
        return None
    x, y = curve
    right = int(np.searchsorted(x, limit, side='right'))
    clipped_x, clipped_y = x[:right], y[:right]
    if clipped_x[-1] < limit:
        value = np.interp(limit, x[right-1:right+1], y[right-1:right+1])
        clipped_x, clipped_y = np.r_[clipped_x, limit], np.r_[clipped_y, value]
    return float(np.clip(np.trapezoid(clipped_y, clipped_x) / limit, 0, 1))


def ranking_ap_f1(labels, scores):
    """Step-wise average precision and best F1 over tied score thresholds.

    AP matches sklearn.metrics.average_precision_score. One sort serves both
    values, which keeps millions of pixels affordable.
    """
    labels = np.asarray(labels, dtype=bool).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.shape != scores.shape:
        raise ValueError('Labels and scores must have the same size.')
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite scores.')
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == labels.size:
        return None, None
    order = np.argsort(-scores, kind='stable')
    ordered = scores[order]
    # A tied score enters the ranking only as one complete block.
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), ordered.size - 1]
    tp = np.cumsum(labels[order], dtype=np.int64)[ends]
    precision = tp / (ends + 1)
    recall = tp / n_pos
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return ap, float(f1.max())


def tpr_at_fpr(labels, scores, fpr=IMAGE_FPR):
    """Best achievable TPR among operating points whose FPR does not exceed fpr."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) != 2:
        return None
    fprs, tprs, _ = roc_curve(labels, scores, drop_intermediate=False)
    return float(tprs[fprs <= fpr + 1e-12].max())


def normal_pixel_threshold(rows, maps, fpr):
    """Pixel threshold from normal test images only; defect masks are not used."""
    if not 0 < fpr < 1:
        raise ValueError('normal_fpr must be in (0,1).')
    normal = np.array([not r['is_anomaly'] for r in rows])
    if not normal.any():
        raise ValueError('Region recall needs at least one normal test image.')
    return float(np.quantile(np.asarray(maps)[normal], 1 - fpr))


def localization_records(rows, maps, masks, threshold, tolerance):
    """Per-image peak hits and per-region detections for anomalous images."""
    images, regions = [], []
    for i, row in enumerate(rows):
        if not row['is_anomaly']:
            continue
        mask, heatmap = masks[i], maps[i]
        record = {'sample_id': row['sample_id'], 'group': row['group'], 'empty_mask': not mask.any(),
                  'peak_distance': None, 'hit_0px': None, 'hit_tol': None, 'n_regions': 0, 'n_detected': 0}
        if record['empty_mask']:
            images.append(record)
            continue
        peak = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        distance = float(ndimage.distance_transform_edt(~mask)[peak])
        labels, count = ndimage.label(mask, structure=STRUCTURE)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
        peaks = np.asarray(ndimage.maximum(heatmap, labels, index=np.arange(1, count + 1)), dtype=np.float64)
        detected = peaks >= threshold
        record.update(peak_distance=distance, hit_0px=distance == 0, hit_tol=distance <= tolerance,
                      n_regions=int(count), n_detected=int(detected.sum()))
        images.append(record)
        for size, hit in zip(sizes, detected):
            regions.append({'sample_id': row['sample_id'], 'group': row['group'], 'region_pixels': int(size),
                            'size_bin': size_bin(int(size)), 'detected': bool(hit)})
    return images, regions


def evaluate_extended(rows, scores, maps, masks, normal_fpr=.01, tolerance=8):
    if tolerance <= 0:
        raise ValueError('tolerance must be positive; the 0 px hit rate is always reported.')
    scores = np.asarray(scores, dtype=np.float64)
    maps, masks = np.asarray(maps), np.asarray(masks, dtype=bool)
    if maps.shape != masks.shape or maps.ndim != 3 or len(rows) != len(maps) or len(scores) != len(rows):
        raise ValueError('rows, scores, maps, and masks disagree in length or shape.')
    if not (np.isfinite(maps).all() and np.isfinite(scores).all()):
        raise ValueError('Non-finite scores or anomaly maps.')
    labels = np.array([r['is_anomaly'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    threshold = normal_pixel_threshold(rows, maps, normal_fpr)
    images, regions = localization_records(rows, maps, masks, threshold, tolerance)
    names = metric_names(tolerance, normal_fpr)
    metrics = {}
    for group in GROUPS:
        selected = np.ones(len(rows), dtype=bool) if group == 'all' else ((groups == group) | (groups == 'normal'))
        positives = selected & (labels == 1)
        in_group = lambda r: group == 'all' or r['group'] == group
        group_images = [r for r in images if in_group(r)]
        valid = [r for r in group_images if not r['empty_mask']]
        group_regions = [r for r in regions if in_group(r)]
        entry = {'n_normal': int((selected & (labels == 0)).sum()), 'n_anomaly': int(positives.sum()),
                 'n_anomaly_empty_transformed_mask': sum(r['empty_mask'] for r in group_images),
                 'n_regions': len(group_regions), **{name: None for name in names}}
        if positives.any():
            current_masks, current_maps = masks[selected], maps[selected]
            curve = pro_curve(current_masks, current_maps)  # one sort serves every FPR limit
            for limit in FPR_LIMITS:
                entry[f'aupro@{limit:g}'] = aupro_from_curve(curve, limit)
            entry['pixel_ap'], entry['pixel_f1max_oracle'] = ranking_ap_f1(current_masks, current_maps)
            entry['image_auroc'] = safe_auroc(labels[selected], scores[selected])
            entry['image_ap'] = ranking_ap_f1(labels[selected], scores[selected])[0]
            entry[f'image_tpr@fpr{IMAGE_FPR:g}'] = tpr_at_fpr(labels[selected], scores[selected])
        if valid:
            entry['pointing@0px'] = float(np.mean([r['hit_0px'] for r in valid]))
            entry[f'pointing@{tolerance:g}px'] = float(np.mean([r['hit_tol'] for r in valid]))
        if group_regions:
            entry[f'region_recall@nfpr{normal_fpr:g}'] = float(np.mean([r['detected'] for r in group_regions]))
        metrics[group] = entry
    result = {'groups': metrics, 'normal_pixel_threshold': threshold, 'normal_fpr': normal_fpr, 'tolerance_px': tolerance}
    return result, images, regions


def paired_bootstrap(images_a, images_b, group, repeats=2000, seed=2026):
    """Stratified paired image bootstrap of B minus A for peak hits and region recall.

    Each model keeps its own fitted normal-image threshold, so the interval is
    conditional on both fitted models and describes test-image sampling only.
    """
    if [r['sample_id'] for r in images_a] != [r['sample_id'] for r in images_b]:
        raise ValueError('Runs have different anomalous images or ordering.')
    keep = [i for i, r in enumerate(images_a)
            if not r['empty_mask'] and (group == 'all' or r['group'] == group)]
    if len(keep) < 10:
        return {'status': 'insufficient_samples', 'n_images': len(keep)}
    a, b = [images_a[i] for i in keep], [images_b[i] for i in keep]
    if [r['n_regions'] for r in a] != [r['n_regions'] for r in b]:
        raise ValueError('Region counts differ; the runs do not share masks.')
    hit_a, hit_b = (np.array([r['hit_tol'] for r in x], dtype=float) for x in (a, b))
    det_a, det_b = (np.array([r['n_detected'] for r in x], dtype=float) for x in (a, b))
    n_regions = np.array([r['n_regions'] for r in a], dtype=float)
    image_groups = np.array([r['group'] for r in a])
    strata = [np.flatnonzero(image_groups == g) for g in ('small', 'large')]
    rng = np.random.default_rng(seed)
    pointing, recall = [], []
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata if len(s)])
        pointing.append(hit_b[idx].mean() - hit_a[idx].mean())
        recall.append((det_b[idx].sum() - det_a[idx].sum()) / n_regions[idx].sum())
    return {'status': 'ok', 'n_images': len(keep),
            'pointing_diff': float(hit_b.mean() - hit_a.mean()),
            'pointing_ci95': np.quantile(pointing, [.025, .975]).tolist(),
            'region_recall_diff': float((det_b.sum() - det_a.sum()) / n_regions.sum()),
            'region_recall_ci95': np.quantile(recall, [.025, .975]).tolist()}


def load_run(folder):
    rows = json.loads((folder / 'samples.json').read_text())
    with np.load(folder / 'predictions.npz', allow_pickle=False) as data:
        if data['sample_ids'].tolist() != [r['sample_id'] for r in rows]:
            raise ValueError(f'Prediction/sample ordering mismatch: {folder}')
        return rows, data['scores'], data['maps'], data['masks']


def budget_of(config):
    return f"bank{config['bank_vectors_requested']}" if config['bank_vectors_requested'] else f"ratio{config['coreset_ratio']:g}"


def aggregate(rows, keys, metrics):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in keys)].append(row)
    summary = []
    for key, values in sorted(buckets.items(), key=lambda item: tuple(map(str, item[0]))):
        seeds = [v['seed'] for v in values]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f'Duplicate seeds in {key}; narrow --root to one study.')
        row = dict(zip(keys, key))
        row.update(n_seeds=len(values), seeds=','.join(map(str, sorted(seeds))),
                   systems=';'.join(sorted({v.get('system', '') for v in values})))
        for metric in metrics:
            data = [v[metric] for v in values if v[metric] is not None]
            row[metric + '_mean'] = float(np.mean(data)) if data else None
            row[metric + '_std'] = float(np.std(data, ddof=1)) if len(data) > 1 else None
        summary.append(row)
    return summary


def robustness(summary, names):
    """Check the report's claims under every metric, using seed means."""
    index = {(r['category'], r['image_size'], r['budget'], r['group'], r['debug'], r['layers']): r for r in summary}
    layer_rows, size_rows = [], []
    for cat, size, budget, group, debug in sorted({k[:5] for k in index}, key=lambda k: tuple(map(str, k))):
        for metric in names:
            cells = {l: index.get((cat, size, budget, group, debug, l)) for l in LAYER_ORDER}
            if any(c is None or c[metric + '_mean'] is None for c in cells.values()):
                continue
            means = {l: cells[l][metric + '_mean'] for l in LAYER_ORDER}
            stds = [cells[l][metric + '_std'] for l in LAYER_ORDER if cells[l][metric + '_std'] is not None]
            layer_rows.append({
                'category': cat, 'image_size': size, 'budget': budget, 'group': group, 'debug': debug, 'metric': metric,
                **means, 'ranking': ' > '.join(sorted(LAYER_ORDER, key=lambda l: -means[l])),
                'layer2_minus_fusion': means['layer2'] - means['fusion'],
                'layer2_minus_layer3': means['layer2'] - means['layer3'],
                'max_seed_std': max(stds) if stds else None,
                'layer2_ge_fusion': means['layer2'] >= means['fusion'],
                'layer2_gt_layer3': means['layer2'] > means['layer3'],
            })
    for cat, budget, group, debug, layer in sorted({(k[0], k[2], k[3], k[4], k[5]) for k in index}, key=lambda k: tuple(map(str, k))):
        low, high = index.get((cat, 224, budget, group, debug, layer)), index.get((cat, 320, budget, group, debug, layer))
        if low is None or high is None:
            continue
        for metric in names:
            if low[metric + '_mean'] is None or high[metric + '_mean'] is None:
                continue
            size_rows.append({'category': cat, 'layers': layer, 'budget': budget, 'group': group, 'debug': debug,
                              'metric': metric, 'size224': low[metric + '_mean'], 'size320': high[metric + '_mean'],
                              'gain_320_minus_224': high[metric + '_mean'] - low[metric + '_mean'],
                              'size320_gt_224': high[metric + '_mean'] > low[metric + '_mean']})
    claims = []
    for metric in names:
        for group in GROUPS:
            for claim, source, field in [('layer2 >= fusion', layer_rows, 'layer2_ge_fusion'),
                                         ('layer2 > layer3', layer_rows, 'layer2_gt_layer3'),
                                         ('320 > 224', size_rows, 'size320_gt_224')]:
                cases = [r for r in source if r['metric'] == metric and r['group'] == group]
                if cases:
                    holds = sum(bool(r[field]) for r in cases)
                    claims.append({'metric': metric, 'group': group, 'claim': claim, 'n_cases': len(cases),
                                   'n_holds': holds, 'holds_fraction': holds / len(cases)})
    return layer_rows, size_rows, claims


def plot_region_recall(summary, output):
    panels = sorted({(r['category'], r['image_size'], r['budget'], r['debug']) for r in summary}, key=lambda k: tuple(map(str, k)))
    for cat, size, budget, debug in panels:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = {}
        for offset, layer in zip((-.08, 0, .08), LAYER_ORDER):
            cells = [r for r in summary if (r['category'], r['image_size'], r['budget'], r['debug'], r['layers']) == (cat, size, budget, debug, layer)]
            cells = sorted(cells, key=lambda r: SIZE_LABELS.index(r['size_bin']))
            if not cells:
                continue
            x = [SIZE_LABELS.index(r['size_bin']) + offset for r in cells]
            ax.errorbar(x, [r['recall_mean'] for r in cells], yerr=[r['recall_std'] or 0 for r in cells],
                        marker='o', capsize=3, label=layer)
            counts.update({r['size_bin']: r['n_regions'] for r in cells})
        ax.set_xticks(range(len(SIZE_LABELS)))
        ax.set_xticklabels([f'{b}\n(n={counts.get(b, 0)})' for b in SIZE_LABELS])
        ax.set(xlabel='Defect region size (transformed-mask pixels)', ylabel='Region recall',
               title=f'{cat}, input {size}, {budget}{" (debug)" if debug else ""}', ylim=(-.03, 1.03))
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'region_recall_{cat}_{size}_{budget}{"_debug" if debug else ""}.png', dpi=160)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/runs'))
    p.add_argument('--output', type=Path, default=Path('results/extended'))
    p.add_argument('--normal-fpr', type=float, default=.01)
    p.add_argument('--tolerance', type=float, default=8)
    p.add_argument('--repeats', type=int, default=2000)
    p.add_argument('--include-debug', action='store_true')
    args = p.parse_args()
    names = metric_names(args.tolerance, args.normal_fpr)
    per_seed, region_rows, images_by_run, hashes = [], [], {}, {}
    for complete in sorted(args.root.rglob('complete.json')):
        folder = complete.parent
        config = json.loads((folder / 'config.json').read_text())
        if config['debug_subset'] and not args.include_debug:
            continue
        rows, scores, maps, masks = load_run(folder)
        result, images, regions = evaluate_extended(rows, scores, maps, masks, args.normal_fpr, args.tolerance)
        key = {'category': config['category'], 'image_size': config['image_size'], 'layers': config['layers'],
               'budget': budget_of(config), 'seed': config['seed'], 'debug': bool(config['debug_subset'])}
        provenance = {'dataset_sha256': config['dataset_sha256'], 'study_source_sha256': config['study_source_sha256'],
                      'system': config['system']}
        for group in GROUPS:
            per_seed.append({**key, 'group': group, **result['groups'][group],
                             'normal_pixel_threshold': result['normal_pixel_threshold'], **provenance})
        run_id = tuple(key.values())
        images_by_run[run_id] = images
        hashes[run_id] = config['dataset_sha256']
        bins = defaultdict(list)
        for region in regions:
            bins[region['size_bin']].append(region['detected'])
        for label in SIZE_LABELS:
            if bins[label]:
                region_rows.append({**key, 'size_bin': label, 'n_regions': len(bins[label]),
                                    'recall': float(np.mean(bins[label])), 'system': config['system']})
        print(f"{folder}: small AUPRO@0.05={result['groups']['small']['aupro@0.05']}", flush=True)
    if not per_seed:
        print('No completed runs found (debug runs need --include-debug).')
        return
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = aggregate(per_seed, ['category', 'image_size', 'layers', 'budget', 'group', 'debug',
                                   'dataset_sha256', 'study_source_sha256'], names)
    region_summary = aggregate(region_rows, ['category', 'image_size', 'layers', 'budget', 'debug', 'size_bin'], ['recall'])
    for row in region_summary:
        row['n_regions'] = next(r['n_regions'] for r in region_rows
                                if all(r[k] == row[k] for k in ('category', 'image_size', 'layers', 'budget', 'debug', 'size_bin')))
    layer_rows, size_rows, claims = robustness(summary, names)
    tol = f'pointing@{args.tolerance:g}px'
    boot_rows = []
    for run_id, images in sorted(images_by_run.items(), key=lambda item: tuple(map(str, item[0]))):
        category, size, layers, budget, seed, debug = run_id
        partner = (category, size, 'fusion', budget, seed, debug)
        if layers != 'layer2' or partner not in images_by_run:
            continue
        if hashes[run_id] != hashes[partner]:
            raise ValueError(f'Dataset hashes differ for {run_id}; pairing is invalid.')
        for group in ('all', 'small'):
            b = paired_bootstrap(images, images_by_run[partner], group, args.repeats)
            boot_rows.append({'category': category, 'image_size': size, 'budget': budget, 'seed': seed, 'debug': debug,
                              'group': group, 'direction': 'fusion minus layer2', 'status': b['status'],
                              'n_images': b['n_images'], f'{tol}_diff': b.get('pointing_diff'),
                              f'{tol}_ci95': b.get('pointing_ci95'),
                              'region_recall_diff': b.get('region_recall_diff'),
                              'region_recall_ci95': b.get('region_recall_ci95')})
    write_csv(output / 'per_seed_extended.csv', per_seed)
    write_csv(output / 'mean_std_extended.csv', summary)
    write_csv(output / 'region_recall_by_size_per_seed.csv', region_rows)
    write_csv(output / 'region_recall_by_size.csv', region_summary)
    write_csv(output / 'robustness_layers.csv', layer_rows)
    write_csv(output / 'robustness_size.csv', size_rows)
    write_csv(output / 'claims.csv', claims)
    write_csv(output / 'bootstrap_layer2_vs_fusion.csv', boot_rows)
    write_json(output / 'extended_summary.json', {
        'settings': {'normal_fpr': args.normal_fpr, 'tolerance_px': args.tolerance, 'repeats': args.repeats,
                     'include_debug': args.include_debug, 'fpr_limits': list(FPR_LIMITS), 'image_fpr': IMAGE_FPR,
                     'size_bins': list(SIZE_LABELS)},
        'notes': ['pixel_f1max_oracle selects its threshold on test masks; use it only as a diagnostic.',
                  'Region recall thresholds come from normal test images only.',
                  'Claims compare three-seed means; check max_seed_std before reading small gaps.'],
        'claims': claims, 'bootstrap': boot_rows})
    plot_region_recall(region_summary, output)
    print(f'Wrote extended metrics for {len(per_seed) // len(GROUPS)} runs to {output}')
    for row in claims:
        if row['group'] == 'small':
            print(f"{row['metric']:<24} {row['claim']:<17} {row['n_holds']}/{row['n_cases']}")


if __name__ == '__main__':
    main()