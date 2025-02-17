import argparse
from joblib import dump, load
from pathlib import Path

import numpy as np
import nibabel as nb
from imops import zoom
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


# ### algorithm hyperparameters ###
M: int = 500
V: float = 0.9999
RS: int = 0xBadFace
# #################################


# ### utils ###
def path_to_resources(is_local: bool = False):
    return Path('/homes/borish/workspace/MOOD_submission_Sample-level_AIRI/resources/') \
        if is_local else Path('/workspace/resources/')


def save_nb_zero_like(fname, path_inp, path_out):
    nii = nb.load(path_inp / fname)
    nb.save(nb.Nifti1Image(np.zeros_like(nii.get_fdata()), affine=nii.affine), path_out / fname)


def clean_embeddings(embeddings: np.ndarray):
    nan_rows = np.sum(np.isnan(embeddings), axis=1).nonzero()[0]
    return np.delete(embeddings, nan_rows, axis=0)


def zoom_image(image: np.ndarray, old_spacing, new_spacing):
    if not isinstance(new_spacing, (tuple, list, np.ndarray)):
        new_spacing = np.broadcast_to(new_spacing, 3)
    scale_factor = np.nan_to_num(np.float32(old_spacing) / np.float32(new_spacing), nan=1)
    return np.array(zoom(image.astype(np.float32), scale_factor, order=1), dtype=image.dtype)


def min_max_scale_q(image: np.ndarray, q_min: float = 0.5, q_max: float = 99.5):
    image = np.clip(image, *np.percentile(np.float32(image), [q_min, q_max]))
    min_val, max_val = image.min(), image.max()
    return np.array((image.astype(np.float32) - min_val) / (max_val - min_val), dtype=image.dtype)


def min_max_scale(x: np.ndarray):
    x = np.copy(x)
    x -= x.min()
    x /= x.max()
    return x
# #############


# ### algorithm ###
def get_embedding(image: np.ndarray, n_bins: int = M):
    return np.histogram(image, bins=n_bins, range=(0, 1), density=True)[0]


def train_and_save_pca(embeddings: np.ndarray, pca_fname: Path, explained_var_ratio: float = V, random_state: int = RS):
    pca_full = PCA(n_components=embeddings.shape[1], random_state=random_state)
    pca_full.fit(embeddings)
    n = min(embeddings.shape[1], np.sum(np.cumsum(pca_full.explained_variance_ratio_) <= explained_var_ratio) + 1)
    print(f'>>> PCA will reduce the dimensions from {embeddings.shape[1]} to {n}.', flush=True)

    pca = PCA(n_components=n, random_state=random_state)
    pca.fit(embeddings)
    dump(pca, pca_fname)

    return pca


def predict_pca(pca, x):
    return pca.transform(x.reshape(1, -1))[0]


def calc_min_distance(x, points):
    return np.sqrt(np.sum((points - x) ** 2, axis=1)).min()


def calc_mah_distance(x, points):
    inv_cov_mahalanobis = np.linalg.inv(np.cov(points.T))
    centered_hist = x - points.mean(axis=0)
    return centered_hist @ inv_cov_mahalanobis @ centered_hist
# #################


def load_and_prep_nifti(fname: str, path_inp, data: str):
    nii = nb.load(path_inp / fname)
    img = nii.get_fdata()

    spacing = np.asarray(nii.header['pixdim'][1:4], dtype='float32')
    new_spacing = (0.7, 0.7875, 0.7) if (data == 'brain') else (0.75, 0.75, 0.8351563)  # if data == 'abdominal'

    return min_max_scale_q(zoom_image(img, old_spacing=spacing, new_spacing=new_spacing))


def predict(fname, path_inp, data: str, pca, embeddings_reduced: np.ndarray, n_bins: int = M):
    image = load_and_prep_nifti(fname=fname, path_inp=path_inp, data=data)
    embedding = predict_pca(pca=pca, x=get_embedding(image, n_bins=n_bins))
    # return calc_min_distance(x=embedding, points=embeddings_reduced)
    return calc_mah_distance(x=embedding, points=embeddings_reduced)


def load_embeddings(path_resources: Path, data: str):
    embeddings = np.load(str(path_resources / f'{data}_embeddings_reduced.npy'))
    return embeddings


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--inp", required=True, type=str)
    parser.add_argument("-o", "--out", required=True, type=str)

    parser.add_argument('-m', "--mode", type=str, default='sample', choices=('pixel', 'sample'), required=False)
    parser.add_argument('-d', '--data', type=str, choices=('brain', 'abdom'), required=True)

    parser.add_argument('--create_embeddings', action='store_true', required=False, default=False)
    parser.add_argument('--local', action='store_true', required=False, default=False)

    args = parser.parse_known_args()[0]

    path_inp = Path(args.inp)
    path_out = Path(args.out)
    path_resources = path_to_resources(is_local=args.local)
    data = args.data

    fnames = [p.name for p in path_inp.glob('*')]

    if args.create_embeddings:
        embeddings_orig = [get_embedding(image=load_and_prep_nifti(fname, path_inp, data), n_bins=M)
                           for fname in tqdm(fnames)]
        embeddings_orig = clean_embeddings(np.stack(embeddings_orig))
        np.save(str(path_resources / f'{data}_embeddings_orig.npy'), np.float32(embeddings_orig))

        pca = train_and_save_pca(embeddings_orig, path_resources / f'{data}_pca.joblib',
                                 explained_var_ratio=V, random_state=RS)
        embeddings_reduced = pca.transform(embeddings_orig)
        np.save(str(path_resources / f'{data}_embeddings_reduced.npy'), np.float32(embeddings_reduced))

    else:
        if args.mode == 'pixel':
            for fname in fnames:
                save_nb_zero_like(fname, path_inp, path_out)

        else:  # args.mode == 'sample':
            embeddings_reduced = np.load(str(path_resources / f'{data}_embeddings_reduced.npy'))
            pca = load(path_resources / f'{data}_pca.joblib')

            scores = []
            for fname in tqdm(fnames):
                try:
                    scores.append(predict(fname, path_inp, data, pca, embeddings_reduced, n_bins=M))
                except Exception:
                    scores.append(0)
            scores = min_max_scale(np.float32(scores))

            for score, fname in zip(scores, fnames):
                with open(str(path_out / (fname + '.txt')), 'w') as wf:
                    wf.write(str(score))


if __name__ == '__main__':
    main()
