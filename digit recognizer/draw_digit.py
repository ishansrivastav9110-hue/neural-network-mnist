import tkinter as tk
import numpy as np
import pickle

# Load trained model
with open("trained_network1.pkl", "rb") as f:
    network = pickle.load(f)


# -----------------------------
# Settings
# -----------------------------

GRID_SIZE = 28
PIXEL_SIZE = 20

WINDOW_SIZE = GRID_SIZE * PIXEL_SIZE


# -----------------------------
# Create window
# -----------------------------

root = tk.Tk()
root.title("MNIST Digit Recognizer")

canvas = tk.Canvas(
    root,
    width=WINDOW_SIZE,
    height=WINDOW_SIZE,
    bg="black"
)

canvas.pack(padx=10, pady=10)


# -----------------------------
# Store actual 28x28 image
# -----------------------------

image = np.zeros((28, 28), dtype=np.float32)


# -----------------------------
# Drawing settings
# -----------------------------

brush_size = 1
erase_mode = False


# -----------------------------
# Draw grid
# -----------------------------

rectangles = []

for row in range(28):

    row_rectangles = []

    for col in range(28):

        x1 = col * PIXEL_SIZE
        y1 = row * PIXEL_SIZE

        x2 = x1 + PIXEL_SIZE
        y2 = y1 + PIXEL_SIZE

        rect = canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            fill="black",
            outline="#222222"
        )

        row_rectangles.append(rect)

    rectangles.append(row_rectangles)



def update_pixel(row, col):

    if row < 0 or row >= 28:
        return

    if col < 0 or col >= 28:
        return

    if erase_mode:
        image[row, col] = 0
    else:
        image[row, col] = 1

    value = int(image[row, col] * 255)

    color = f"#{value:02x}{value:02x}{value:02x}"

    canvas.itemconfig(
        rectangles[row][col],
        fill=color
    )

def draw(event):

    col = event.x // PIXEL_SIZE
    row = event.y // PIXEL_SIZE

    radius = brush_size // 2

    for r in range(row - radius, row + radius + 1):

        for c in range(col - radius, col + radius + 1):

            update_pixel(r, c)


canvas.bind("<Button-1>", draw)
canvas.bind("<B1-Motion>", draw)

def set_brush_size(value):

    global brush_size

    brush_size = int(value)


brush_label = tk.Label(
    root,
    text="Brush size"
)

brush_label.pack()


brush_slider = tk.Scale(
    root,
    from_=1,
    to=5,
    orient=tk.HORIZONTAL,
    command=set_brush_size
)

brush_slider.set(1)
brush_slider.pack()


def toggle_eraser():

    global erase_mode

    erase_mode = not erase_mode

    if erase_mode:
        erase_button.config(text="Eraser: ON")
    else:
        erase_button.config(text="Eraser: OFF")


erase_button = tk.Button(
    root,
    text="Eraser: OFF",
    command=toggle_eraser,
    width=12
)

erase_button.pack(pady=3)

def clear():

    global image

    image = np.zeros((28, 28), dtype=np.float32)

    for row in range(28):

        for col in range(28):

            canvas.itemconfig(
                rectangles[row][col],
                fill="black"
            )

    result_label.config(text="Prediction: -")

clear_button = tk.Button(
    root,
    text="Clear",
    command=clear,
    width=12
)
clear_button.pack(pady=3)

def predict():

    input_image = image.flatten()

    output = network.forward(input_image)

    digit = np.argmax(output)

    confidence = output[digit] * 100

    result_label.config(
        text=f"Prediction: {digit}  ({confidence:.2f}%)"
    )

    print("Prediction:", digit)
    print("Confidence:", confidence)
    print("Probabilities:", output)


predict_button = tk.Button(
    root,
    text="Predict",
    command=predict,
    width=12
)

predict_button.pack(pady=5)

result_label = tk.Label(
    root,
    text="Prediction: -",
    font=("Arial", 20)
)

result_label.pack(pady=10)

root.mainloop()