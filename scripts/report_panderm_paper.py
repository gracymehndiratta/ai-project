"""Summarize the original-model reference without conflating metrics or splits."""
from pathlib import Path
import json


def table(frame):
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for _,row in frame.iterrows():
        lines.append('| '+' | '.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def main():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parent.parent
    out=root/'outputs/diagnostics/panderm_paper_reference'
    stats=json.loads((out/'metrics.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    verification=json.loads((out/'verification.json').read_text())
    base=json.loads((root/'outputs/feature_models/panderm_base_comparison/mlp/metrics.json').read_text())
    rows=[{'experiment':'Published Large + probe','test_population':'Manuscript HAM10000','macro_f1':'Not reported in cited table',
           'weighted_f1':.926,'balanced_accuracy':.807}]
    for title,metric,population in [
        ('Released Large + author probe',stats['released_test'],'Released 1232-image test'),
        ('Released Large + author probe',stats['official_test'],'Official ISIC 1512-image test'),
        ('Previous Base + custom NN',base['test'],'Our 1492-image internal test'),
        ('Previous Base + custom NN',base['official_test'],'Official ISIC 1512-image test')]:
        rows.append({'experiment':title,'test_population':population,
            **{key:metric[key] for key in ['macro_f1','weighted_f1','balanced_accuracy']}})
    comparison=pd.DataFrame(rows)
    comparison.to_csv(out/'paper_comparison.csv',index=False)
    per_class=[]
    for split,metric in stats.items():
        for name,row in metric['per_class'].items():
            if name in ['akiec','bcc','bkl','df','mel','nv','vasc']:
                per_class.append({'split':split,'lesion':name,**row})
    pd.DataFrame(per_class).to_csv(out/'per_class_statistics.csv',index=False)
    fig,axis=plt.subplots(figsize=(10,4))
    values=[.926,stats['released_test']['weighted_f1'],stats['official_test']['weighted_f1'],base['official_test']['weighted_f1']]
    axis.bar(['Paper\nHAM weighted F1','Released Large\nReleased test','Released Large\nOfficial ISIC test','Base + our NN\nOfficial ISIC test'],values,
        color=['#666666','#2563eb','#14b8a6','#a855f7'])
    axis.set(ylabel='Weighted F1',ylim=(0,1),title='Different test populations and training protocols: not a single leaderboard')
    for index,value in enumerate(values):axis.text(index,value+.01,f'{value:.1%}',ha='center')
    fig.tight_layout();fig.savefig(out/'weighted_f1_comparison.png',dpi=160);plt.close(fig)
    lines=['# Correction and original PanDerm reference', '',
        '**The earlier 40.60% macro-F1 row was a custom nearest-neighbor diagnostic, not the published PanDerm model. It cannot be used to judge whether the research result reproduces.** Previous raw outputs are preserved.', '',
        '## What differed', '',
        '- Original research backbone: PanDerm Large (24 blocks, 1024-dimensional features). Previous local comparison: smaller PanDerm Base (12 blocks, 768-dimensional features).',
        '- Published HAM10000 classifier: logistic-regression linear probing on frozen features. Previous local reference substituted nearest-neighbor retrieval; the other rows used custom trained heads.',
        '- Published headline HAM10000 result: weighted F1 0.926 and balanced accuracy0.807 in Supplementary Table4. Our headline was macro F1 on a different official ISIC test set.',
        '- The paper describes a cleaned60/20/20 split. The downloadable CSV actually has train8207,val575,test1232 and one trainisic row. These released split labels were preserved, not repaired or replaced.', '',
        '## Original-model reference performed', '',
        'Used the official Large checkpoint and the pinned, unedited author encoder implementation. Every required encoder tensor was strictly verified; unused pretraining decoder tensors were excluded, as in the official inference builder. Used the authors\' exact logistic_regression.py class with C=71.68, max_iter1000, seed0, and their training-row shuffle seed42. No custom scaling, class balancing, smoothing, SMOTE or hyperparameter tuning was introduced. The backbone weights were frozen throughout.', '',
        'The existing local JPEG files for all 10015 released CSV rows matched the downloadable archive by byte size and CRC32. Training used exactly8207 train rows;575 val rows and1232 test rows were kept separate. The one trainisic row was ignored by the same equality filters as the upstream Dataset. Used partitions have no lesion overlap.', '',
        'Only hardware/environment adaptations were used: batch8 for extraction, float32 with TF32 disabled, and the already installed PyTorch/timm versions. This is a released-code reference attempt, not a claim of bitwise reproduction of the manuscript experiment.', '',
        '## Results', '',table(comparison), '',
        'The official-test rows use the same external images, but the Large reference trains on8207 images using the published fixed probe, whereas our Base heads trained on7014 using custom validation-selected heads. This is not a controlled architecture-only ablation.', '',
        '## All reference splits', '',table(pd.read_csv(out/'comparison.csv')), '',
        f'All {verification["previous_artifacts_unchanged"]} previous model artifacts are unchanged. Seven independent real-image embedding checks passed. Model/probe hashes and the exact released split are recorded in protocol.json, verification.json and released_split_manifest.csv.', '',
        'The previous nearest-neighbor result should remain only as a separately labelled diagnostic. The absence of a classification head in the released pretrained encoder does not mean the paper tested an encoder alone: the paper explicitly trains a task-specific probe. The remaining gap to a published result cannot be assigned to one cause from these experiments, especially while the released CSV differs from the manuscript description.', '',
        '## Primary sources', '',
        '- [Nature Medicine paper and Methods](https://www.nature.com/articles/s41591-025-03747-y)',
        '- [Supplementary Table4](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41591-025-03747-y/MediaObjects/41591_2025_3747_MOESM1_ESM.pdf)',
        '- [Official release and code](https://github.com/SiyuanYan1/PanDerm)', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print(table(comparison))


if __name__=='__main__':main()
