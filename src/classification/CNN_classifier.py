from __future__ import annotations

# ... existing code ...

import os
import logging
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import tensorflow as tf
from tensorflow import keras #type: ignore
from omegaconf import DictConfig

from utils.utils import get_logger, ensure_dir
from src.preprocessing.base_preprocessor import BasePreprocessor
from src.preprocessing.CNN_preprocessor import CNNPreprocessor 
from src.classification.base_classifier import BaseClassifier


class CNNClassifier(BaseClassifier):
    """Convolutional Neural Network classifier for metro line digit recognition.

    The classifier is designed for recognising digits 1-14 extracted from metro line
    pictograms. The network is intentionally lightweight so that it can be trained on
    a small imbalanced dataset. Class imbalance is handled by *class weights* that can
    be supplied during training.
    """

    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """Create a CNNClassifier instance.

        Parameters
        ----------
        cfg : DictConfig
            Configuration containing hyper-parameters such as *input_shape*, *epochs*,
            *batch_size*, *model_path*, etc.
        logger : logging.Logger, optional
            Custom logger. If *None*, a default project-wide logger is provided.
        """
        super().__init__(cfg, logger)

        self.input_shape: Tuple[int, int, int] = tuple(cfg.get("input_shape", [64, 64, 1]))  # type: ignore[arg-type]
        self.epochs: int = int(cfg.get("epochs", 50))
        self.batch_size: int = int(cfg.get("batch_size", 32))
        self.learning_rate: float = float(cfg.get("learning_rate", 1e-3))
        self.model_path: str = cfg.get("model_path", "models/cnn_classifier.h5")
        self._model: Optional[keras.Model] = None

        self.class_labels: List[int] = list(range(1, 15))
        self.label_to_index: Dict[int, int] = {lbl: idx for idx, lbl in enumerate(self.class_labels)}
        self.index_to_label: Dict[int, int] = {idx: lbl for lbl, idx in self.label_to_index.items()}
        self.num_classes: int = len(self.class_labels)


    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:  # type: ignore[override]
        """Train the CNN.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training images and integer labels (1-14).
        X_val, y_val : np.ndarray, optional
            Validation images/labels. If *None* validation is skipped.
        class_weight : dict, optional
            Mapping from *class index* (0-13) to weight.

        Returns
        -------
        dict
            Keras history.history dictionary.
        """
        y_train_idx = np.vectorize(self.label_to_index.get)(y_train)
        y_val_idx: Optional[np.ndarray] = None
        if y_val is not None and len(y_val) > 0:
            y_val_idx = np.vectorize(self.label_to_index.get)(y_val)

        if self._model is None:
            _model = self._build_model()
            _model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            self._model = _model

        model = self._model 

        history = model.fit( #type: ignore
            X_train,
            y_train_idx,
            validation_data=(X_val, y_val_idx) if X_val is not None else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            verbose=1,
        )

        if self.model_path:
            ensure_dir(os.path.dirname(self.model_path))
            self._model.save(self.model_path) #type: ignore
            self.logger.info(f"CNN model saved to {self.model_path}")

        return history.history  # type: ignore[attr-defined]

    def predict(self, image: np.ndarray) -> Tuple[int, float]:  # type: ignore[override]
        """Predict digit for a single ROI.

        Parameters
        ----------
        image : np.ndarray
            ROI image in BGR/Gray format.

        Returns
        -------
        class_id : int
            Predicted metro line (1-14). **-1** indicates rejection.
        confidence : float
            Softmax probability of the predicted class.
        """
        if self._model is None:
            # Lazy load
            self.load(self.model_path)
        if isinstance(self.preprocessor, CNNPreprocessor):
            image = self.preprocessor.preprocess(image)
        else:
            raise ValueError("CNNClassifier requires a CNNPreprocessor to be set")

        image_batch = np.expand_dims(image, axis=0)  # (1, H, W, C)
        probs: np.ndarray = self._model.predict(image_batch, verbose=0)[0] #type: ignore
        idx: int = int(np.argmax(probs))
        confidence: float = float(probs[idx])
        class_id: int = self.index_to_label[idx]
        return class_id, confidence

    def save(self, path: str) -> None:  # type: ignore[override]
        ensure_dir(os.path.dirname(path))
        if self._model is None:
            self.logger.warning("No model to save – skip")
            return
        self._model.save(path)
        self.logger.info(f"CNN model saved to {path}")

    def load(self, path: str) -> None:  # type: ignore[override]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self._model = keras.models.load_model(path)
        self.logger.info(f"CNN model loaded from {path}")

    def _build_model(self) -> keras.Model:
        """Define the CNN architecture."""
        inputs = keras.Input(shape=self.input_shape)
        x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)

        x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)

        x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_digit_classifier")
        self.logger.info(model.summary())
        return model