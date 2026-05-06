import numpy as np

from sklearn.datasets import load_breast_cancer


_BREAST_CANCER_SHIFT_SPECS = (
    # feature index, class label to perturb, sign (+1 increase, -1 decrease),
    # base scale in units of that class-conditional std.
    (1, 0, -1.0, 0.60),   # mean texture
    (23, 0, +1.0, 0.35),  # worst area
    (26, 0, -1.0, 0.45),  # worst concavity / convexity proxy in existing notebook
    (29, 0, -1.0, 0.55),  # worst fractal dimension
)

def small_random_shifts(shifted_ds, og_ds, feature_inds, class_inds):
    if type(class_inds) == int:
        feature_data = og_ds[class_inds, feature_inds]
        std_og = np.std(og_ds[class_inds, feature_inds], axis=0)
        random_texture = np.random.normal(0, std_og/25, feature_data.shape)
        shifted_ds[class_inds,feature_inds] = np.abs(feature_data + random_texture)
    else:
        feature_data = og_ds[np.ix_(class_inds, feature_inds)]
        std_og = np.std(og_ds[np.ix_(class_inds, feature_inds)], axis=0)
        random_texture = np.random.normal(0, std_og/25, feature_data.shape)
        shifted_ds[np.ix_(class_inds,feature_inds)] = np.abs(feature_data + random_texture)

    return shifted_ds

def perform_shift(shifted_ds, og_ds, feature_ind, class_inds, inc, sf, print_values=False, feature_name=None):
    feature_data = og_ds[class_inds, feature_ind]
    std_og = np.std(feature_data)
    random_texture = np.random.normal(std_og/sf, std_og/(sf*2), feature_data.shape)

    if not inc:
        random_texture = random_texture * -1
    shifted_ds[class_inds, feature_ind] = feature_data + random_texture


    if print_values:
        max_og = np.max(feature_data)
        min_og = np.min(feature_data)

        new_min = np.min(shifted_ds[class_inds, feature_ind])
        new_max = np.max(shifted_ds[class_inds, feature_ind])
        new_std = np.std(shifted_ds[class_inds, feature_ind])

        print(f'Original Min {feature_name}: {min_og}\nShifted Min {feature_name}: {new_min}\n')
        print(f'Original Max {feature_name}: {max_og}\nShifted Max {feature_name}: {new_max}\n')
        print(f'Original STD {feature_name}: {std_og}\nShifted STD {feature_name}: {new_std}\n')

def perform_noise_injection(shifted_ds, og_ds, feature_inds, sample, print_values=False, feature_name=None):
    feature_data = og_ds[sample, feature_inds]
    std_og = np.std(og_ds[:, feature_inds], axis=0)
    random_texture = np.random.normal(0, std_og/10, feature_data.shape)

    shifted_ds[sample, feature_inds] = np.abs(feature_data + random_texture)

def load_covariate_shifts():
    data = load_breast_cancer()
    X, y = data.data, data.target

    malignant_inds = np.where(y == 0)[0]
    benign_inds = np.where(y == 1)[0]
    covar_shift_dataset = np.array([row for row in X]) # avoid mutability issues with python

    # randomly decrease malignant textures
    perform_shift(covar_shift_dataset, X, 1, malignant_inds, False, 2, feature_name='Texture Mean')

    # randomly increase worst malignant areas
    perform_shift(covar_shift_dataset, X, 23, malignant_inds, True, 12, feature_name='Worst Area')

    # randomly decrease worst malignant convexity
    perform_shift(covar_shift_dataset, X, 26, malignant_inds, False, 8, feature_name='Worst Convexity')

    # randomly decrease worst fractal distance
    perform_shift(covar_shift_dataset, X, 29, malignant_inds, False, 6, feature_name='Worst Fractal Distance')

    small_random_shifts(covar_shift_dataset, X, np.arange(30), benign_inds)
    covar_features = [1, 23, 26, 29]
    inds = np.array([i for i in range(30) if i not in covar_features])
    small_random_shifts(covar_shift_dataset, X, inds, malignant_inds)

    return covar_shift_dataset, y

def load_noise_injection():
    data = load_breast_cancer()
    X, y = data.data, data.target

    feature_noise_dataset = np.array([row for row in X])

    for i in range(feature_noise_dataset.shape[0]):
        features = np.random.choice(np.arange(30), 3)
        perform_noise_injection(feature_noise_dataset, X, features, i)
        others = np.array([i for i in range(30) if i not in features])
        small_random_shifts(feature_noise_dataset, X, others, i)

    feature_data = X[:, 2]
    max_og = np.max(feature_data)
    min_og = np.min(feature_data)
    std_og = np.std(feature_data)

    new_min = np.min(feature_noise_dataset[:, 2])
    new_max = np.max(feature_noise_dataset[:, 2])
    new_std = np.std(feature_noise_dataset[:, 2])
    feature_name = 'wha'
    print(f'Original Min {feature_name}: {min_og}\nShifted Min {feature_name}: {new_min}\n')
    print(f'Original Max {feature_name}: {max_og}\nShifted Max {feature_name}: {new_max}\n')
    print(f'Original STD {feature_name}: {std_og}\nShifted STD {feature_name}: {new_std}\n')

    return feature_noise_dataset, y


def make_matched_breast_cancer_shift(
    X: np.ndarray,
    y: np.ndarray,
    severity: float = 1.0,
    random_state: int | None = 42,
    jitter_scale: float = 0.05,
) -> np.ndarray:
    """
    Create a matched covariate-shifted copy of the provided Breast Cancer data.

    The shift is applied to the *same examples* so clean-vs-shifted comparisons
    are not confounded by different sample composition. Severity scales a set of
    domain-inspired perturbations on malignant examples and adds mild background
    jitter to the remaining features.
    """
    if severity < 0:
        raise ValueError("severity must be non-negative.")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel().astype(int)
    if X.ndim != 2 or X.shape[1] != 30:
        raise ValueError("Expected Breast Cancer features with shape (n_samples, 30).")
    if len(y) != X.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    rng = np.random.default_rng(random_state)
    shifted = X.copy()

    for feature_idx, class_label, direction, base_scale in _BREAST_CANCER_SHIFT_SPECS:
        class_mask = y == class_label
        feature_values = X[class_mask, feature_idx]
        feature_std = np.std(feature_values)
        if feature_std == 0:
            continue
        mean_shift = direction * severity * base_scale * feature_std
        noise = rng.normal(
            loc=mean_shift,
            scale=max(1e-12, 0.20 * severity * feature_std),
            size=feature_values.shape,
        )
        shifted[class_mask, feature_idx] = np.maximum(0.0, feature_values + noise)

    targeted_features = {spec[0] for spec in _BREAST_CANCER_SHIFT_SPECS}
    other_features = np.array([idx for idx in range(X.shape[1]) if idx not in targeted_features])
    if len(other_features) > 0 and jitter_scale > 0:
        feature_std = np.std(X[:, other_features], axis=0)
        background_noise = rng.normal(
            loc=0.0,
            scale=jitter_scale * severity * np.maximum(feature_std, 1e-12),
            size=(X.shape[0], len(other_features)),
        )
        shifted[:, other_features] = np.maximum(0.0, shifted[:, other_features] + background_noise)

    return shifted


def sweep_matched_shift_severities(
    X: np.ndarray,
    y: np.ndarray,
    severities: list[float] | np.ndarray,
    random_state: int | None = 42,
    jitter_scale: float = 0.05,
) -> dict[float, np.ndarray]:
    """
    Build matched shifted copies of the same evaluation set at multiple severities.
    """
    shifts: dict[float, np.ndarray] = {}
    for idx, severity in enumerate(severities):
        shifts[float(severity)] = make_matched_breast_cancer_shift(
            X,
            y,
            severity=float(severity),
            random_state=None if random_state is None else random_state + idx,
            jitter_scale=jitter_scale,
        )
    return shifts

if __name__ == '__main__':
    load_noise_injection()
