"""Checks for the post-hoc extended metrics used by the Colab notebook."""
import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from extended_metrics import (aupro_from_curve, evaluate_extended, normal_pixel_threshold,
                              paired_bootstrap, ranking_ap_f1, size_bin, tpr_at_fpr)
from study.metrics import aupro, pro_curve


def toy(size=16, n_normal=3):
    rows = [{'sample_id': f'good/{i}', 'is_anomaly': 0, 'group': 'normal'} for i in range(n_normal)]
    masks = [np.zeros((size, size), bool) for _ in range(n_normal)]
    small = np.zeros((size, size), bool)
    small[2:4, 2:4] = True
    large = np.zeros((size, size), bool)
    large[8:14, 8:14] = True
    large[0, 15] = True  # a second, separate region
    rows += [{'sample_id': 'bad/small', 'is_anomaly': 1, 'group': 'small'},
             {'sample_id': 'bad/large', 'is_anomaly': 1, 'group': 'large'}]
    masks += [small, large]
    return rows, np.stack(masks)


class ExtendedMetrics(unittest.TestCase):
    def test_pixel_ap_matches_sklearn_with_ties(self):
        rng = np.random.default_rng(0)
        labels = rng.random(500) < .1
        scores = np.round(rng.random(500) + labels * .3, 1)
        ap, _ = ranking_ap_f1(labels, scores)
        self.assertAlmostEqual(ap, average_precision_score(labels, scores), places=12)

    def test_f1_max_known_value(self):
        _, f1 = ranking_ap_f1([1, 0, 1, 0], [.9, .8, .7, .1])
        self.assertAlmostEqual(f1, .8)

    def test_single_class_is_undefined(self):
        self.assertEqual(ranking_ap_f1([0, 0], [.1, .2]), (None, None))
        self.assertIsNone(tpr_at_fpr([1, 1], [.1, .2]))

    def test_perfect_maps(self):
        rows, masks = toy()
        maps = masks.astype(np.float32)
        scores = np.array([r['is_anomaly'] for r in rows], float)
        result, _, _ = evaluate_extended(rows, scores, maps, masks)
        for group in ('all', 'small', 'large'):
            entry = result['groups'][group]
            for key in ('aupro@0.01', 'pixel_ap', 'pixel_f1max_oracle', 'image_auroc', 'image_ap',
                        'image_tpr@fpr0.05', 'pointing@0px', 'pointing@8px', 'region_recall@nfpr0.01'):
                self.assertAlmostEqual(entry[key], 1.0, msg=f'{group}/{key}')
        self.assertEqual(result['groups']['large']['n_regions'], 2)

    def test_inverted_maps_miss_every_region(self):
        rows, masks = toy()
        maps = 1 - masks.astype(np.float32)
        result, _, _ = evaluate_extended(rows, np.zeros(len(rows)), maps, masks)
        entry = result['groups']['all']
        self.assertEqual(entry['pointing@0px'], 0.0)
        self.assertEqual(entry['region_recall@nfpr0.01'], 0.0)

    def test_aupro_matches_study_implementation(self):
        rows, masks = toy()
        maps = np.random.default_rng(1).random(masks.shape).astype(np.float32)
        result, _, _ = evaluate_extended(rows, np.zeros(len(rows)), maps, masks)
        curve = pro_curve(masks, maps)
        for limit in (.3, .05, .01):
            self.assertAlmostEqual(result['groups']['all'][f'aupro@{limit:g}'], aupro(masks, maps, limit), places=12)
            self.assertAlmostEqual(aupro_from_curve(curve, limit), aupro(masks, maps, limit), places=12)

    def test_empty_transformed_mask_is_counted_not_scored(self):
        rows, masks = toy()
        rows.append({'sample_id': 'bad/vanished', 'is_anomaly': 1, 'group': 'small'})
        masks = np.concatenate([masks, np.zeros((1,) + masks.shape[1:], bool)])
        maps = masks.astype(np.float32)
        result, images, _ = evaluate_extended(rows, np.zeros(len(rows)), maps, masks)
        small = result['groups']['small']
        self.assertEqual(small['n_anomaly'], 2)
        self.assertEqual(small['n_anomaly_empty_transformed_mask'], 1)
        self.assertEqual(small['pointing@0px'], 1.0)
        self.assertTrue(images[-1]['empty_mask'])

    def test_size_bin_edges(self):
        expected = {1: '<64', 63: '<64', 64: '64-255', 255: '64-255', 256: '256-1023',
                    1023: '256-1023', 1024: '>=1024'}
        for pixels, label in expected.items():
            self.assertEqual(size_bin(pixels), label)

    def test_threshold_ignores_defect_images(self):
        rows, masks = toy()
        maps = np.random.default_rng(2).random(masks.shape)
        before = normal_pixel_threshold(rows, maps, .01)
        maps[[r['is_anomaly'] == 1 for r in rows]] *= 100
        self.assertEqual(before, normal_pixel_threshold(rows, maps, .01))

    def test_pointing_tolerance(self):
        rows = [{'sample_id': 'good', 'is_anomaly': 0, 'group': 'normal'},
                {'sample_id': 'bad', 'is_anomaly': 1, 'group': 'small'}]
        masks = np.zeros((2, 32, 32), bool)
        masks[1, 10, 10] = True
        maps = np.zeros((2, 32, 32))
        maps[1, 10, 15] = 1.0  # peak 5 px from the defect
        result, _, _ = evaluate_extended(rows, np.array([0., 1.]), maps, masks, tolerance=8)
        self.assertEqual(result['groups']['small']['pointing@0px'], 0.0)
        self.assertEqual(result['groups']['small']['pointing@8px'], 1.0)

    def test_tpr_at_low_fpr(self):
        labels = [0] * 20 + [1] * 5
        scores = list(range(20)) + [30, 31, 32, 33, 34]
        self.assertEqual(tpr_at_fpr(labels, scores, .05), 1.0)
        scores[0] = 40  # one normal image now outranks every defect
        self.assertEqual(tpr_at_fpr(labels, scores, .05), 1.0)  # FPR 1/20 is still allowed
        scores[1] = 41
        self.assertEqual(tpr_at_fpr(labels, scores, .05), 0.0)

    def test_bootstrap_of_identical_runs_is_zero(self):
        images = [{'sample_id': str(i), 'group': 'small' if i % 3 else 'large', 'empty_mask': False,
                   'hit_tol': i % 2 == 0, 'n_regions': 2, 'n_detected': i % 3} for i in range(12)]
        result = paired_bootstrap(images, images, 'all', repeats=200)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['pointing_diff'], 0.0)
        self.assertEqual(result['pointing_ci95'], [0.0, 0.0])
        self.assertEqual(result['region_recall_ci95'], [0.0, 0.0])

    def test_bootstrap_rejects_mismatched_runs(self):
        a = [{'sample_id': 'x', 'group': 'small', 'empty_mask': False, 'hit_tol': True, 'n_regions': 1, 'n_detected': 1}]
        b = [dict(a[0], sample_id='y')]
        with self.assertRaises(ValueError):
            paired_bootstrap(a, b, 'all')


if __name__ == '__main__':
    unittest.main()