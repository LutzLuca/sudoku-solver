import cv2 as cv
import numpy as np
from typing import Tuple, Optional
import sys
import os

KERNEL: np.ndarray = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

Point = Tuple[int, int]
Rect = Tuple[Point, Point]
MatLike = cv.typing.MatLike


def preprocess(img: MatLike) -> MatLike:
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    thres = cv.adaptiveThreshold(
        blurred.astype(np.uint8),
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        11,
        3,
    )
    return cv.dilate(thres, KERNEL)


def square_image(img: MatLike) -> MatLike:
    sz = max(img.shape[:2])
    adjusted_sz = sz + (9 - sz % 9) % 9
    return cv.resize(img, (adjusted_sz, adjusted_sz))


def centre_to_size(img: MatLike, size: int, margin: int = 0) -> MatLike:
    h, w = img.shape[:2]

    def centre_pad(length: int) -> Tuple[int, int]:
        pad = (size - length) // 2
        return pad, pad + (length % 2)

    if h > w:
        ratio = (size - margin) / h
        w, h = int(ratio * w), int(ratio * h)
        l_pad, r_pad = centre_pad(w)
        t_pad = b_pad = margin // 2
    else:
        ratio = (size - margin) / w
        w, h = int(ratio * w), int(ratio * h)
        t_pad, b_pad = centre_pad(h)
        l_pad = r_pad = margin // 2

    img = cv.resize(img, (w, h))
    img = cv.copyMakeBorder(
        img, t_pad, b_pad, l_pad, r_pad, cv.BORDER_CONSTANT, value=0
    )
    return cv.resize(img, (size, size))


def find_sudoku_grid(img: MatLike) -> np.ndarray:
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)

    bottom_right = max(largest_contour, key=lambda p: p[0][0] + p[0][1])[0]
    top_left = min(largest_contour, key=lambda p: p[0][0] + p[0][1])[0]
    bottom_left = min(largest_contour, key=lambda p: p[0][0] - p[0][1])[0]
    top_right = max(largest_contour, key=lambda p: p[0][0] - p[0][1])[0]

    return np.array([top_left, top_right, bottom_right, bottom_left])


def find_largest_feature(
    image: MatLike, tl: Point, br: Point
) -> Optional[Tuple[MatLike, Rect]]:
    cpy = image.copy()
    h, w = image.shape[:2]
    max_area, seed = -1, (None, None)

    for x in range(tl[0], br[0]):
        for y in range(tl[1], br[1]):
            if cpy[y, x] == 255:
                area, _, _, _ = cv.floodFill(cpy, None, (x, y), 100)
                if area > max_area:
                    max_area = area
                    seed = (x, y)

    if any(c is None for c in seed):
        return None

    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = 4 | cv.FLOODFILL_MASK_ONLY | 255 << 8
    cv.floodFill(cpy, mask, seed, 255, flags=flags)

    img = cv.bitwise_and(image, image, mask=mask[1 : h + 1, 1 : w + 1])

    coords = np.column_stack(np.where(img == 255))
    if coords.size == 0:
        return None

    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)

    return (img, ((top_left[1], top_left[0]), (bottom_right[1], bottom_right[0])))


def warp_perspective(rect: np.ndarray, grid: MatLike) -> MatLike:
    tl, tr, br, bl = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(br - tr), np.linalg.norm(bl - tl)))

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(grid, M, (width, height), flags=cv.INTER_LINEAR)
    return square_image(warped)


def extract(image_path: str) -> np.ndarray:
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    preprocessed_image = preprocess(image)
    grid_points = find_sudoku_grid(preprocessed_image)
    cropped = warp_perspective(
        np.array(grid_points, dtype="float32"), preprocessed_image
    )

    sq_sz = cropped.shape[0] // 9
    margin = int(sq_sz / 2.5)
    tl, br = (margin, margin), (sq_sz - margin, sq_sz - margin)

    def cut_out_digit(res: Optional[Tuple[MatLike, Rect]]) -> MatLike:
        if res is None:
            return np.zeros((28, 28), dtype=np.uint8)

        digit, bbox = res
        w, h = bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]

        if (w * h) > 100:
            return centre_to_size(cut_out_rect(digit, bbox), 28, 4)
        return np.zeros((28, 28), dtype=np.uint8)

    return np.fromiter(
        map(
            cut_out_digit,
            (find_largest_feature(sq, tl, br) for sq in raw_squares(cropped, sq_sz)),
        ),
        dtype=np.ndarray,
        count=81,
    )


def raw_squares(img: MatLike, sq_sz: int):
    for row in range(9):
        for col in range(9):
            yield cut_out_rect(
                img,
                ((col * sq_sz, row * sq_sz), ((col + 1) * sq_sz, (row + 1) * sq_sz)),
            )


def cut_out_rect(img: MatLike, rect: Rect) -> MatLike:
    return img[int(rect[0][1]) : int(rect[1][1]), int(rect[0][0]) : int(rect[1][0])]


def grid_image(digits: np.ndarray, color: int = 255) -> MatLike:
    rows = [
        np.concatenate(
            [
                cv.copyMakeBorder(
                    img.copy(), 1, 1, 1, 1, cv.BORDER_CONSTANT, None, color
                )
                for img in digits[i * 9 : (i + 1) * 9]
            ],
            axis=1,
        )
        for i in range(9)
    ]
    return np.concatenate(rows)


def main(image_path: str):
    digits = extract(image_path)
    grid = grid_image(digits)

    cv.imshow("result", grid)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extractor.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        sys.exit(1)

    main(image_path)
