const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadButton = document.getElementById("upload-button");
const statusDisplay = document.getElementById("status");
const sudokuGrid = document.getElementById("sudoku-grid");

uploadButton.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    statusDisplay.textContent = "Uploading and solving...";

    try {
        const resized = await resizeImage(file);
        const formData = new FormData();
        formData.append("file", resized);

        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });

        const result = await response.json();

        if (result.success) {
            statusDisplay.textContent = "Sudoku solved successfully!";
            displaySudoku(result.solved_grid);
        } else {
            statusDisplay.textContent = `Error: ${result.error}`;
        }
    } catch (error) {
        statusDisplay.textContent = `Error: ${error.message}`;
    }
});

async function resizeImage(imgFile) {
    // If the image resolution is too high
    // the digit extraction does not work properly
    const MAX_WIDTH = 800;
    const MAX_HEIGHT = 800;

    return new Promise((res, rej) => {
        const reader = new FileReader();
        reader.onload = () => {
            const img = new Image();

            img.onload = () => {
                const canvas = document.createElement("canvas");
                const ctx = canvas.getContext("2d");
                let width = img.width;
                let height = img.height;

                if (width > height) {
                    if (width > MAX_WIDTH) {
                        height = Math.round((height * MAX_WIDTH) / width);
                        width = MAX_WIDTH;
                    }
                } else {
                    if (height > MAX_HEIGHT) {
                        width = Math.round((width * MAX_HEIGHT) / height);
                        height = MAX_HEIGHT;
                    }
                }

                canvas.width = width;
                canvas.height = height;
                ctx.drawImage(img, 0, 0, width, height);
                canvas.toBlob(res, imgFile.type);
            };

            img.onerror = rej;
            img.src = reader.result;
        };

        reader.onerror = rej;
        reader.readAsDataURL(imgFile);
    });
}

function displaySudoku(grid) {
    sudokuGrid.innerHTML = "";
    grid.flat().forEach((num) => {
        const cell = document.createElement("div");
        cell.className = "cell";
        cell.textContent = num !== 0 ? num : "";
        sudokuGrid.appendChild(cell);
    });
}
