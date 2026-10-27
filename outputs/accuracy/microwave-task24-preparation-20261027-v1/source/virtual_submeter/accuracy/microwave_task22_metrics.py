"""Independent tied-score AP; no protocol, model or data dependencies."""
import numpy as np

def independent_ap(scores,truth):
    values,inverse=np.unique(scores,return_inverse=True)
    pos=np.bincount(inverse,weights=np.asarray(truth,np.int64),minlength=len(values))[::-1]
    total=np.bincount(inverse,minlength=len(values))[::-1]
    tp=np.cumsum(pos);predicted=np.cumsum(total)
    return float(np.sum(pos/max(int(np.sum(truth)),1)*tp/np.maximum(predicted,1)))

