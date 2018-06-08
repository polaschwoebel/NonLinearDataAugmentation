import numpy as np
from scipy import stats


def all_classes_present(labels):
    nr_imgs = labels.shape[0]
    for img_indx in range(nr_imgs):
        if len(np.unique(labels[img_indx,:,:,:])) != 135:
            return False
        else:
            return True


# Note: implementation for the case where all classes are present
# (check function above).
def mean_dice(pred, true):
    num_classes = max(len(np.unique(pred)), len(np.unique(true)))
    D = np.zeros(num_classes)
    for i in range(0, num_classes):
        pos_true = (true == i)
        pos_pred = (pred == i)
        A = np.sum(pos_pred)
        B = np.sum(pos_true)
        A_B = np.sum(np.logical_and(pos_true, pos_pred))
        D[i] = (2 * A_B) / (A + B)
    return (1 / num_classes) * np.sum(D)


def pairwise_distances(labels):
    nr_imgs = labels.shape[0]
    pairwise_distances = np.zeros((nr_imgs, nr_imgs))
    for ix1 in range(nr_imgs):
        for ix2 in range(ix1 , nr_imgs):
            pairwise_distances[ix1, ix2] = mean_dice(labels[ix1,:,:,:], labels[ix2,:,:,:])


def get_quantile(pairwise_distances):
    pairwise_flat = pairwise_distances.flatten()
    nonzero_dice = pairwise_flat[pairwise_flat < 1]
    nonzero_dice = nonzero_dice[nonzero_dice > 0]
    quant = stats.mstats.mquantiles(nonzero_dice, prob=[0.15], alphap=0.4, betap=0.4, axis=None, limit=())
    return quant[0]


def make_pairs_list(pairwise_distances, quant):
    relevant_pairs = np.logical_and(pairwise_distances > quant, pairwise_distances < 1)
    indices = np.argwhere(relevant_pairs)
    np.savetxt('pairs.csv', indices, delimiter=' ', fmt='%10.0f', newline='\n', header='', footer='', comments='# ', encoding=None)
