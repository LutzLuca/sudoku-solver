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


def prepare_data_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    y_train_inds = np.where(y_train != 0)
    y_test_inds = np.where(y_test != 0)

    x_train, x_test = x_train[y_train_inds], x_test[y_test_inds]
    y_train, y_test = y_train[y_train_inds], y_test[y_test_inds]

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, num_classes=10)[:, 1:]
    y_test = keras.utils.to_categorical(y_test, num_classes=10)[:, 1:]

    return x_train, x_test, y_train, y_test


def prepare_data_fonts(
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


def prepare_data(data_dir: str):
    x_train_mnist, x_test_mnist, y_train_mnist, y_test_mnist = prepare_data_mnist()
    x_train_fonts, x_test_fonts, y_train_fonts, y_test_fonts = prepare_data_fonts(
        data_dir
    )

    return (
        np.concatenate((x_train_fonts, x_train_mnist)),
        np.concatenate((x_test_fonts, x_test_mnist)),
        np.concatenate((y_train_fonts, y_train_mnist)),
        np.concatenate((y_test_fonts, y_test_mnist)),
    )
