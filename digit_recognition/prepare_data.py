from pathlib import Path
import cv2 as cv
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple


def load_images(data_dir: str) -> Dict[int, List[np.ndarray]]:
    def normalize_image(img: np.ndarray) -> np.ndarray:
        return np.expand_dims(img / 255.0, -1)

    image_dict: Dict[int, List[np.ndarray]] = {}
    for num in range(1, 10):
        digit_dir = Path(data_dir) / str(num)
        image_dict[num] = [
            normalize_image(cv.imread(str(img_path), cv.IMREAD_GRAYSCALE))
            for img_path in digit_dir.glob("*.webp")
        ]

    return image_dict


def prepare_data_mnist(mnist_percentile: float = 1.0):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    y_train_inds = np.where(y_train != 0)
    y_test_inds = np.where(y_test != 0)

    x_train, x_test = x_train[y_train_inds], x_test[y_test_inds]
    y_train, y_test = y_train[y_train_inds], y_test[y_test_inds]

    train_size = int(len(x_train) * mnist_percentile)
    test_size = int(len(x_test) * mnist_percentile)

    x_train = x_train[:train_size]
    y_train = y_train[:train_size]
    x_test = x_test[:test_size]
    y_test = y_test[:test_size]

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train - 1, num_classes=9)
    y_test = keras.utils.to_categorical(y_test - 1, num_classes=9)

    return x_train, x_test, y_train, y_test


def prepare_data_fonts(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_dict = load_images(data_dir)

    x = np.array([img for images in image_dict.values() for img in images])
    y = np.array(
        [np.repeat(digit - 1, len(images)) for digit, images in image_dict.items()]
    )
    y = keras.utils.to_categorical(y.reshape(-1), num_classes=9)

    return train_test_split(x, y, test_size=0.15, shuffle=True, random_state=42)


def prepare_data(data_dir: str, mnist_percentile: float = 1.0):
    print("Preparing data...")
    x_train_mnist, x_test_mnist, y_train_mnist, y_test_mnist = prepare_data_mnist(
        mnist_percentile
    )
    x_train_fonts, x_test_fonts, y_train_fonts, y_test_fonts = prepare_data_fonts(
        data_dir
    )

    x_train = np.concatenate((x_train_fonts, x_train_mnist))
    x_test = np.concatenate((x_test_fonts, x_test_mnist))
    y_train = np.concatenate((y_train_fonts, y_train_mnist))
    y_test = np.concatenate((y_test_fonts, y_test_mnist))

    train_indices = np.arange(len(x_train))
    test_indices = np.arange(len(x_test))
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]

    return x_train, x_test, y_train, y_test
