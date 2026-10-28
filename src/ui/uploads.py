"""Image validation and an explicitly experimental in-distribution screen."""
from io import BytesIO
import warnings
import numpy as np
import pandas as pd
from PIL import Image,ImageOps,UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits


def decode_image(payload):
    if not payload: raise ValueError('Choose an image file first.')
    if len(payload)>10*1024*1024: raise ValueError('Use an image smaller than 10 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as candidate:
                candidate.verify()
            with Image.open(BytesIO(payload)) as candidate:
                image=ImageOps.exif_transpose(candidate).convert('RGB')
                image.thumbnail((4096,4096))
                return image.copy()
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning) as error:
        raise ValueError('This file is not a readable, safe image. Try a JPG or PNG.') from error


def quality_issue(image):
    values=np.asarray(image.resize((64,64)),dtype=float)
    if values.std()<2 or np.max(values.std(axis=(0,1)))<2:
        return 'The image is nearly uniform or blank; no useful image structure is visible.'
    if max(image.size)/min(image.size)>4:
        return 'The image shape is unsuitable. Upload an uncropped dermoscopic photograph.'
    return None


class SimilarityScreen:
    """One-class cosine screen; does not diagnose lesion presence or healthy skin."""
    def __init__(self,backbone):
        if backbone=='panderm_large':
            path='data/features/panderm_large_72dfc47de5e585b2.npy'
            manifest='outputs/feature_models/panderm_large_comparison/split_manifest.csv'
        elif backbone=='panderm_base':
            path='data/features/panderm_base_9e9b688f78eb850d.npy'
            manifest='outputs/feature_models/panderm_base_comparison/split_manifest.csv'
        else:
            raise ValueError('Similarity screening is available for PanDerm classifiers only.')
        frame=pd.read_csv(manifest);values=np.load(path,mmap_mode='r')
        rows=np.flatnonzero(frame.split=='train');subset=frame.iloc[rows]
        reference,calibration=next(StratifiedGroupKFold(3,shuffle=True,random_state=2026).split(rows,subset.label,subset.lesion_id))
        assert set(subset.iloc[reference].lesion_id).isdisjoint(set(subset.iloc[calibration].lesion_id))
        self.mean=np.asarray(values[rows[reference]]).mean(0,dtype=np.float64).astype(np.float32)
        self.reference=self.normalize(np.asarray(values[rows[reference]])-self.mean)
        held=self.normalize(np.asarray(values[rows[calibration]])-self.mean)
        scores=[]
        with threadpool_limits(limits=4):
            for begin in range(0,len(held),128): scores.extend((held[begin:begin+128]@self.reference.T).max(1).tolist())
        self.threshold=float(np.quantile(scores,.01))
        self.reference_count=len(reference);self.calibration_count=len(calibration)
        self.training_images=len(rows)
        self.calibration_image_ids=subset.iloc[calibration].image_id.tolist()
        self.reference_image_ids=subset.iloc[reference].image_id.tolist()

    @staticmethod
    def normalize(values):
        return values/np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-8)

    def check(self,feature):
        value=self.normalize(np.asarray(feature,dtype=np.float32).reshape(1,-1)-self.mean)
        score=float((value@self.reference.T).max())
        return {'accepted':score>=self.threshold,'similarity':score,'threshold':self.threshold,
            'interpretation':'Experimental training-image similarity, not a probability of having a lesion.'}
