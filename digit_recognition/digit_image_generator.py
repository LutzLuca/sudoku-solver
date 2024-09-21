from PIL import Image, ImageDraw, ImageFont
from PIL.ImageOps import invert
from glob import glob
import os
from argparse import ArgumentParser

GRAY_SCALE_MODE = "L"


def generate_digit_images(fonts_folder: str, output_path: str):
    font_paths = glob(os.path.join(fonts_folder, "*.ttf"))

    for num in range(1, 10):
        os.makedirs(os.path.join(output_path, str(num)), exist_ok=True)

    for font_path in font_paths:
        font_name, _ = os.path.splitext(os.path.basename(font_path))

        try:
            font = ImageFont.truetype(font_path, 24)

            for num in range(1, 10):
                output_file = os.path.join(output_path, str(num), f"{font_name}.webp")
                create_digit_image(num, font, output_file)

        except OSError as e:
            print(f"Error processing font {font_name}: {e}")


def create_digit_image(
    digit: int, font: ImageFont.FreeTypeFont, output_path: str, size=(28, 28)
):
    image = Image.new(GRAY_SCALE_MODE, size, color=255)
    draw = ImageDraw.Draw(image)

    draw.text(
        (size[0] / 2, size[1] / 2),
        text=str(digit),
        font=font,
        anchor="mm",
        align="center",
    )

    inverted = invert(image)
    inverted.save(output_path, format="webp")


def main():
    parser = ArgumentParser()
    parser.add_argument("fonts_folder", help="Path to the folder containing fonts")
    parser.add_argument(
        "output_path", help="Path to the output folder for generated images"
    )

    args = parser.parse_args()
    fonts_folder, output_path = args.fonts_folder, args.output_path

    if not os.path.isdir(fonts_folder):
        raise ValueError(f"Invalid fonts folder: {fonts_folder} is not a directory")

    generate_digit_images(fonts_folder, output_path)


if __name__ == "__main__":
    main()
