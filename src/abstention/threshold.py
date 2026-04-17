import numpy as np


class ThresholdAbstainer:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def abstain_mask(self, probs: np.ndarray) -> np.ndarray:
        """
        probs: shape (n_samples, n_classes)
        returns: boolean mask, True where we abstain
        """
        confidence = probs.max(axis=1)
        return confidence < self.threshold

    def apply(self, probs: np.ndarray):
        """
        Returns:
            preds: predicted labels, with -1 for abstained examples
            abstain_mask: True where abstained
            confidence: max predicted probability
        """
        preds = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        abstain_mask = confidence < self.threshold

        preds_with_abstain = preds.copy()
        preds_with_abstain[abstain_mask] = -1

        return preds_with_abstain, abstain_mask, confidence