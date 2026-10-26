"""Verify the continued optimizer chain and summarize exactly three new updates."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def main():
    os.chdir(ROOT)
    import json
    import numpy as np
    import pandas as pd
    from src.models.panderm import sha256_file
    out=Path('outputs/diagnostics/panderm_test_adaptation_steps3_5');out.mkdir(exist_ok=True)
    rows=[]
    for step in range(2,6):
        folder=Path(f'outputs/diagnostics/panderm_test_adaptation_step{step}')
        metrics=json.loads((folder/'metrics.json').read_text())
        receipt=json.loads((folder/'update_receipt.json').read_text())
        verification=json.loads((folder/'verification.json').read_text())
        assert verification['passed'] and verification['cumulative_optimizer_updates']==step
        assert receipt['optimizer_updates']==1 and set(receipt['optimizer_state_steps'])=={step}
        assert receipt['learning_rate']==.00028 and receipt['images_seen']==2744
        assert receipt['adapted_model_sha256']==sha256_file(folder/'adapted/model.joblib')
        if step>2:
            previous=Path(f'outputs/diagnostics/panderm_test_adaptation_step{step-1}')
            assert receipt['source_model_sha256']==sha256_file(previous/'adapted/model.joblib')
            previous_metrics=json.loads((previous/'metrics.json').read_text())
            for partition in metrics:
                for key in ['accuracy','macro_f1','macro_f05']:
                    np.testing.assert_allclose(metrics[partition]['before'][key],previous_metrics[partition]['after'][key],atol=1e-12)
            hashes=json.loads((folder/'previous_artifact_hashes.json').read_text())
            assert all(sha256_file(path)==digest for path,digest in hashes.items())
        combined=metrics['combined_adaptation']['after']
        rows.append({'cumulative_updates':step,'new_updates_this_request':max(0,step-2),
            'combined_fitting_accuracy':combined['accuracy']*100,'combined_fitting_macro_f1':combined['macro_f1']*100,
            'combined_fitting_macro_f05':combined['macro_f05']*100,
            'HAM_validation_macro_f1':metrics['validation']['after']['macro_f1']*100,
            'official_validation_macro_f1':metrics['official_validation']['after']['macro_f1']*100})
    result=pd.DataFrame(rows);result.to_csv(out/'comparison.csv',index=False)
    def table(columns):
        data=result[columns]
        text=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
        for row in data.itertuples(index=False,name=None):
            text.append('| '+' | '.join(f'{value:.2f}' if isinstance(value,float) else str(value) for value in row)+' |')
        return '\n'.join(text)
    text='''# Three further adaptation iterations: steps 3, 4 and 5

Exactly three additional full-combined-data gradient updates at learning rate0.00028 were applied to the previous step2 checkpoint. AdamW moments and counters are continued, not reset. Each update accumulates gradients over all2,744 examples across22 microbatches before one optimizer.step. Frozen PanDerm Large, original training-only scaling, original class weights and the128-unit head architecture are preserved. Every earlier checkpoint is unchanged. The latest state is step5; there is no rollback or test-score-based checkpoint selection.

## Combined-image fitting scores (percent)

'''+table(['cumulative_updates','combined_fitting_accuracy','combined_fitting_macro_f1','combined_fitting_macro_f05'])+'''

## Validation excluded from these updates (percent)

'''+table(['cumulative_updates','HAM_validation_macro_f1','official_validation_macro_f1'])+'''

The combined images are training data for these copies; their scores are fitting diagnostics, not held-out performance. Validation images were excluded from the adaptation updates, but their scores were inspected previously, so they are not fresh independent confirmation. Official validation is only193 images, including one dermatofibroma and three vascular cases; macro F1 can change sharply when very few rare-class predictions change. Do not interpret the large step4 macro-F1 change as robust evidence of a comparable population-wide improvement.

Separate checkpoints, optimizer states, metrics, image-level predictions and before/after reports are saved in the adjacent panderm_test_adaptation_step3, step4 and step5 folders. Source-checkpoint hashes, optimizer counters, before/after metric continuity and preservation hashes were independently verified for the chain.
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    (out/'verification.json').write_text(json.dumps({'passed':True,'additional_updates':3,'cumulative_updates':5,
        'learning_rate':.00028,'optimizer_state_continued':True,'earlier_artifacts_unchanged':True},indent=2))
    print(result.to_string(index=False),flush=True)


if __name__=='__main__': main()
