import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow import keras
from argparse import ArgumentParser
from prepare_data import prepare_data


def create_model() -> Sequential:
    model = Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(9, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


def train_model(
    save_path: str, batch_size: int, epochs: int, data_dir: str, mnist: float
) -> None:
    save_path = os.path.splitext(save_path)[0] + ".keras"

    x_train, x_val, y_train, y_val = prepare_data(data_dir, mnist)
    model = create_model()

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
    )

    model.save(save_path)
    print(f"Model saved at: {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a digit recognition model")
    parser.add_argument(
        "--save_path",
        default="models/model.keras",
        type=str,
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", default=10, type=int, help="Number of epochs for training"
    )
    parser.add_argument(
        "--data_dir",
        default="./data/digits",
        type=str,
        help="Directory containing the training data",
    )
    parser.add_argument("--mnist", default=1.0, type=float)

    train_model(**vars(parser.parse_args()))
