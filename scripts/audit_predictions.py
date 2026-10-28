"""Recalculate saved results against independent labels and declared exposure."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.prediction_audit import audit_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--exposure', required=True, help='Union of all training, tuning and selection cohorts')
    parser.add_argument('--reported-metrics')
    parser.add_argument('--split')
    parser.add_argument('--require-disjoint', action='store_true')
    parser.add_argument('--output', required=True, help='New JSON receipt; existing files are never overwritten')
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('Output already exists; choose a new receipt path')
    try:
        result = audit_predictions(args.predictions, args.truth, args.exposure,
                                   args.reported_metrics, args.split, args.require_disjoint)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('Prediction audit saved. This receipt alone does not establish independent generalization.')


if __name__ == '__main__':
    main()
