"""Measure the existing linear configuration on the new tuning folds."""
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from src.data.full_isic import CLASS_NAMES
    from src.utils.feature_training import score_predictions
    from src.utils.tuned_training import class_weights
    out=Path('outputs/feature_models/panderm_large_refined')
    frame=pd.read_csv('outputs/feature_models/panderm_large_comparison/split_manifest.csv')
    folds=pd.read_csv(out/'training_folds.csv')
    rows=np.flatnonzero(frame.split=='train')
    x=np.array(np.load('data/features/panderm_large_72dfc47de5e585b2.npy',mmap_mode='r')[rows])
    y=frame.iloc[rows].label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
    probability=np.zeros((len(y),7))
    details=[]
    for fold in range(3):
        held=folds.cv_fold.to_numpy()==fold
        scaler=StandardScaler().fit(x[~held])
        model=LogisticRegression(C=.01,class_weight=dict(enumerate(class_weights(y[~held],.5))),max_iter=2000,random_state=42)
        with threadpool_limits(limits=4): model.fit(scaler.transform(x[~held]),y[~held])
        probability[held]=model.predict_proba(scaler.transform(x[held]))
        metric=score_predictions(y[held],probability[held])
        details.append({'fold':fold,'macro_f1':metric['macro_f1'],'accuracy':metric['accuracy']})
    metric=score_predictions(y,probability)
    np.save(out/'linear_reference_oof.npy',probability)
    result={'reference':'Existing linear config C=.01,weight power=.5, train-fold scaling; diagnostic only, not an additional final model',
        'folds':details,'oof_macro_f1':metric['macro_f1'],'oof_accuracy':metric['accuracy']}
    (out/'linear_reference_cv.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
