from argparse import ArgumentParser
import cv2 as cv
from .extractor import grid_image, extract_digits
import numpy as np
from .solver import solve
import tensorflow as tf


def process_sudoku_image(model, img):
    digits = extract_digits(img)

    grid = np.zeros(81, dtype=np.uint8)
    for idx, digit in filter(lambda x: x[1] is not None, enumerate(digits)):
        normalized_digit = digit / 255.0
        reshaped_digit = normalized_digit.reshape(28, 28, 1)
        input_digit = np.expand_dims(reshaped_digit, axis=0)

        pred_labels = model.predict(input_digit, verbose=0)
        pred_label = np.argmax(pred_labels, axis=1)[0] + 1
        grid[idx] = pred_label

    solved = solve(grid)
    if not solved:
        raise Exception("Failed to solve Sukodu")

    return grid


def main(model_path: str, img_path: str):
    model = tf.keras.models.load_model(model_path)

    try:
        grid = process_sudoku_image(model, img_path)
        digit_imgs = np.array([create_digit_image(digit) for digit in grid])

        cv.imshow("Grid", grid_image(digit_imgs))
        cv.waitKey(0)
        cv.destroyAllWindows()
    except Exception as e:
        print(str(e))


def create_digit_image(digit: int, size=(28, 28)):
    img = np.zeros(size, dtype=np.uint8)
    if digit == 0:
        return img

    digit = str(digit)
    (text_w, text_h), _ = cv.getTextSize(digit, cv.FONT_HERSHEY_SIMPLEX, 1, 2)

    img_w, img_h = img.shape[1], img.shape[0]
    x = (img_w - text_w) // 2
    y = (img_h + text_h) // 2

    cv.putText(img, digit, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    return img


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("img")
    parser.add_argument("--model", default="./models/model.keras")

    args = parser.parse_args()
    main(args.model, args.img)
