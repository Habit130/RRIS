import argparse
from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate RIS masks from prediction and GT directories.'
    )
    parser.add_argument(
        '--pred_dir',
        required=True,
        help='prediction mask directory; files are matched recursively by relative path stem',
    )
    parser.add_argument(
        '--gt_dir',
        required=True,
        help='ground-truth mask directory; files are matched recursively by relative path stem',
    )
    parser.add_argument(
        '--threshold',
        default=0.5,
        type=float,
        help='threshold for binarizing prediction masks when they are not already binary',
    )
    return parser.parse_args()


def _validate_threshold(threshold):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('--threshold must be within [0, 1], got {}'.format(threshold))


def _collect_mask_files(root_dir):
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError('Mask directory not found: {}'.format(root))

    file_map = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        relative_path = path.relative_to(root)
        if any(part.startswith('.') for part in relative_path.parts):
            continue

        relative_stem = relative_path.with_suffix('').as_posix()
        if relative_stem in file_map:
            raise ValueError(
                'Duplicate mask key [{}] under {}. Rename files to make relative stems unique.'
                .format(relative_stem, root)
            )
        file_map[relative_stem] = path

    if not file_map:
        raise ValueError('No supported mask files found under {}'.format(root))
    return root, file_map


def _load_grayscale_mask(mask_path):
    with Image.open(mask_path) as image:
        return np.array(image.convert('L'))


def _binarize_prediction(pred_array, threshold):
    pred_float = pred_array.astype(np.float32)

    # Treat masks with values outside [0, 1] as 8-bit images/probability maps stored in [0, 255].
    if pred_float.max() > 1.0 or pred_float.min() < 0.0:
        pred_float = pred_float / 255.0

    return pred_float >= threshold


def _binarize_ground_truth(gt_array):
    return gt_array.astype(np.float32) > 0


def _compute_sample_metrics(pred_mask, gt_mask):
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    intersection = int(np.logical_and(pred_bool, gt_bool).sum())
    pred_area = int(pred_bool.sum())
    gt_area = int(gt_bool.sum())
    union = pred_area + gt_area - intersection

    if union == 0:
        iou = 1.0
    else:
        iou = intersection / float(union)

    denom_dice = pred_area + gt_area
    if denom_dice == 0:
        dice = 1.0
    else:
        dice = (2.0 * intersection) / float(denom_dice)

    return {
        'intersection': intersection,
        'union': union,
        'iou': iou,
        'dice': dice,
    }


def evaluate_masks(pred_dir, gt_dir, threshold):
    pred_root, pred_files = _collect_mask_files(pred_dir)
    gt_root, gt_files = _collect_mask_files(gt_dir)

    pred_keys = set(pred_files.keys())
    gt_keys = set(gt_files.keys())

    missing_gt = sorted(pred_keys - gt_keys)
    ignored_gt = sorted(gt_keys - pred_keys)

    if missing_gt:
        raise ValueError(
            'Missing GT masks for {} prediction masks. First 10 missing keys: {}'.format(
                len(missing_gt), missing_gt[:10]
            )
        )

    sample_keys = sorted(pred_files.keys())
    if not sample_keys:
        raise ValueError('No matched mask pairs found between {} and {}'.format(pred_root, gt_root))

    per_sample_iou = []
    per_sample_dice = []
    total_intersection = 0
    total_union = 0
    precision_hits = {0.5: 0, 0.7: 0, 0.9: 0}

    for key in sample_keys:
        pred_array = _load_grayscale_mask(pred_files[key])
        gt_array = _load_grayscale_mask(gt_files[key])

        if pred_array.shape != gt_array.shape:
            raise ValueError(
                'Shape mismatch for [{}]: prediction {} vs GT {}'.format(
                    key, pred_array.shape, gt_array.shape
                )
            )

        pred_mask = _binarize_prediction(pred_array, threshold)
        gt_mask = _binarize_ground_truth(gt_array)
        sample_metrics = _compute_sample_metrics(pred_mask, gt_mask)

        per_sample_iou.append(sample_metrics['iou'])
        per_sample_dice.append(sample_metrics['dice'])
        total_intersection += sample_metrics['intersection']
        total_union += sample_metrics['union']

        for precision_threshold in precision_hits:
            precision_hits[precision_threshold] += int(sample_metrics['iou'] >= precision_threshold)

    sample_count = len(sample_keys)
    results = {
        'num_samples': sample_count,
        'ignored_gt': len(ignored_gt),
        'mIoU': float(np.mean(per_sample_iou)),
        'oIoU': 1.0 if total_union == 0 else total_intersection / float(total_union),
        'Dice': float(np.mean(per_sample_dice)),
        'P@0.5': precision_hits[0.5] / float(sample_count),
        'P@0.7': precision_hits[0.7] / float(sample_count),
        'P@0.9': precision_hits[0.9] / float(sample_count),
    }
    return results


def _format_results_table(results, threshold, pred_dir, gt_dir):
    rows = [
        ('Pred Dir', str(Path(pred_dir).expanduser().resolve())),
        ('GT Dir', str(Path(gt_dir).expanduser().resolve())),
        ('Threshold', '{:.3f}'.format(threshold)),
        ('Samples', str(results['num_samples'])),
        ('Ignored GT', str(results['ignored_gt'])),
        ('mIoU', '{:.2f}'.format(results['mIoU'] * 100.0)),
        ('oIoU', '{:.2f}'.format(results['oIoU'] * 100.0)),
        ('Dice', '{:.2f}'.format(results['Dice'] * 100.0)),
        ('P@0.5', '{:.2f}'.format(results['P@0.5'] * 100.0)),
        ('P@0.7', '{:.2f}'.format(results['P@0.7'] * 100.0)),
        ('P@0.9', '{:.2f}'.format(results['P@0.9'] * 100.0)),
    ]

    key_width = max(len(key) for key, _ in rows)
    value_width = max(len(value) for _, value in rows)
    border = '+-{}-+-{}-+'.format('-' * key_width, '-' * value_width)

    lines = [border]
    for key, value in rows:
        lines.append('| {:{}} | {:{}} |'.format(key, key_width, value, value_width))
    lines.append(border)
    return '\n'.join(lines)


def main():
    args = parse_args()
    _validate_threshold(args.threshold)

    try:
        results = evaluate_masks(args.pred_dir, args.gt_dir, args.threshold)
    except Exception as exc:
        raise SystemExit('Evaluation failed: {}'.format(exc))

    print(_format_results_table(results, args.threshold, args.pred_dir, args.gt_dir))


if __name__ == '__main__':
    main()
