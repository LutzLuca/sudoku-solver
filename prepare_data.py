from pathlib import Path
import cv2 as cv
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple


def normalize_image(img: np.ndarray) -> np.ndarray:
    return np.expand_dims(img / 255.0, -1)


def load_images(data_dir: str) -> Dict[int, List[np.ndarray]]:
    image_dict: Dict[int, List[np.ndarray]] = {}

    for num in range(1, 10):
        digit_dir = Path(data_dir) / str(num)
        image_dict[num] = [
            normalize_image(cv.imread(str(img_path), cv.IMREAD_GRAYSCALE))
            for img_path in digit_dir.glob("*.webp")
        ]

    return image_dict


def prepare_data(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("Preparing data...")
    image_dict = load_images(data_dir)

    x = np.array([img for images in image_dict.values() for img in images])
    y = np.array(
        [np.repeat(digit, len(images)) for digit, images in image_dict.items()]
    )
    y = keras.utils.to_categorical(y.reshape(-1), num_classes=10)[:, 1:]

    return train_test_split(x, y, test_size=0.15, shuffle=True, random_state=42)
