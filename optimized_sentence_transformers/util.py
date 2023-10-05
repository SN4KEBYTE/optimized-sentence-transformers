import numpy as np


def cos_sim(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    if len(a.shape) == 1:
        a = a[np.newaxis, :]

    if len(b.shape) == 1:
        b = b[np.newaxis, :]

    return dot_score(
        normalize_embeddings(a),
        normalize_embeddings(b),
    )


def dot_score(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    if len(a.shape) == 1:
        a = a[np.newaxis, :]

    if len(b.shape) == 1:
        b = b[np.newaxis, :]

    return np.matmul(
        a,
        np.transpose(
            b,
            (
                1,
                0,
            ),
        ),
    )


def pairwise_cos_sim(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    return pairwise_dot_score(
        normalize_embeddings(a),
        normalize_embeddings(b),
    )


def pairwise_dot_score(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    return (a * b).sum(axis=-1)


def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    return embeddings / np.linalg.norm(
        embeddings,
        axis=1,
    ).reshape((-1, 1))
