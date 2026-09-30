import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Configuration
# ============================================================

NETWORK_FILE = "training/mnist_train.bin"

MNIST_IMAGES = "mnist/t10k-images.idx3-ubyte"
MNIST_LABELS = "mnist/t10k-labels.idx1-ubyte"

INPUT_SIZE = 784
OUTPUT_SIZE = 10
NUM_CLASSES = 10

NETWORK_VERSION = 4

OUTPUT_DIRECTORY = "mnist_results"

N_WORST_ERRORS = 20
N_WORST_CONFIDENT_ERRORS = 20


# ============================================================
# Activation types
# ============================================================

RELU = 0
SOFTMAX = 1


# ============================================================
# Utility functions
# ============================================================

def read_exact(f, size):
    data = f.read(size)

    if len(data) != size:
        raise EOFError(
            f"Unexpected end of file: expected {size} bytes, "
            f"got {len(data)}"
        )

    return data


def read_u32(f):
    return struct.unpack("<I", read_exact(f, 4))[0]


def read_u64(f):
    return struct.unpack("<Q", read_exact(f, 8))[0]


def read_doubles(f, count):
    if count == 0:
        return np.empty(0, dtype=np.float64)

    data = read_exact(f, count * 8)

    return np.frombuffer(
        data,
        dtype="<f8"
    ).copy()


# ============================================================
# MNIST loader
# ============================================================

def load_mnist_images(filename):
    with open(filename, "rb") as f:

        magic = struct.unpack(">I", read_exact(f, 4))[0]

        if magic != 2051:
            raise ValueError(
                f"Invalid MNIST image file magic: {magic}"
            )

        count = struct.unpack(">I", read_exact(f, 4))[0]
        rows = struct.unpack(">I", read_exact(f, 4))[0]
        cols = struct.unpack(">I", read_exact(f, 4))[0]

        if rows != 28 or cols != 28:
            raise ValueError(
                f"Expected 28x28 images, got {rows}x{cols}"
            )

        data = read_exact(
            f,
            count * rows * cols
        )

        images = np.frombuffer(
            data,
            dtype=np.uint8
        ).reshape(
            count,
            rows,
            cols
        )

        return images.copy()


def load_mnist_labels(filename):
    with open(filename, "rb") as f:

        magic = struct.unpack(">I", read_exact(f, 4))[0]

        if magic != 2049:
            raise ValueError(
                f"Invalid MNIST label file magic: {magic}"
            )

        count = struct.unpack(">I", read_exact(f, 4))[0]

        data = read_exact(f, count)

        labels = np.frombuffer(
            data,
            dtype=np.uint8
        )

        return labels.copy()


# ============================================================
# Network
# ============================================================

class Layer:

    def __init__(
        self,
        n_neurons,
        n_inputs,
        activation,
        weights,
        biases,
        mts,
        vts,
        bias_mts,
        bias_vts
    ):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.activation = activation

        self.weights = weights
        self.biases = biases

        self.mts = mts
        self.vts = vts

        self.bias_mts = bias_mts
        self.bias_vts = bias_vts

    def forward(self, inputs):

        W = self.weights.reshape(
            self.n_neurons,
            self.n_inputs
        )

        z = W @ inputs + self.biases

        if self.activation == RELU:

            return np.maximum(
                0.0,
                z
            )

        elif self.activation == SOFTMAX:

            z_max = np.max(z)

            exp_z = np.exp(
                z - z_max
            )

            denominator = np.sum(exp_z)

            return exp_z / denominator

        else:

            raise ValueError(
                f"Unknown activation type: "
                f"{self.activation}"
            )


class Network:

    def __init__(
        self,
        input_size,
        loss_type,
        adam_t,
        layers
    ):
        self.input_size = input_size
        self.loss_type = loss_type
        self.adam_t = adam_t
        self.layers = layers

    def forward(self, inputs):

        a = inputs

        for layer in self.layers:
            a = layer.forward(a)

        return a


# ============================================================
# Network loader
# ============================================================

def load_network(filename):

    print(f"Loading network: {filename}")

    with open(filename, "rb") as f:

        # ----------------------------------------------------
        # Magic
        # ----------------------------------------------------

        magic = read_exact(f, 3)

        if magic != b"FNN":
            raise ValueError(
                f"Invalid network magic: {magic}"
            )

        # ----------------------------------------------------
        # Version
        # ----------------------------------------------------

        version = read_u32(f)

        if version != NETWORK_VERSION:
            raise ValueError(
                f"Unsupported network version: {version}. "
                f"Expected {NETWORK_VERSION}"
            )

        # ----------------------------------------------------
        # Network metadata
        # ----------------------------------------------------

        input_size = read_u64(f)
        n_layers = read_u64(f)

        loss_type = read_u32(f)
        adam_t = read_u64(f)

        if input_size != INPUT_SIZE:
            raise ValueError(
                f"Expected input size {INPUT_SIZE}, "
                f"got {input_size}"
            )

        print(f"Version:       {version}")
        print(f"Input size:    {input_size}")
        print(f"Layers:        {n_layers}")
        print(f"Loss type:     {loss_type}")
        print(f"Adam timestep: {adam_t}")

        # ----------------------------------------------------
        # Layers
        # ----------------------------------------------------

        layers = []

        previous_size = input_size

        for layer_index in range(n_layers):

            n_neurons = read_u64(f)
            activation = read_u32(f)
            n_inputs = read_u32(f)

            if n_inputs != previous_size:
                raise ValueError(
                    f"Layer {layer_index}: expected "
                    f"{previous_size} inputs, "
                    f"got {n_inputs}"
                )

            if activation not in (RELU, SOFTMAX):
                raise ValueError(
                    f"Layer {layer_index}: "
                    f"unknown activation {activation}"
                )

            weight_count = n_neurons * n_inputs

            weights = read_doubles(
                f,
                weight_count
            )

            biases = read_doubles(
                f,
                n_neurons
            )

            mts = read_doubles(
                f,
                weight_count
            )

            vts = read_doubles(
                f,
                weight_count
            )

            bias_mts = read_doubles(
                f,
                n_neurons
            )

            bias_vts = read_doubles(
                f,
                n_neurons
            )

            layer = Layer(
                n_neurons,
                n_inputs,
                activation,
                weights,
                biases,
                mts,
                vts,
                bias_mts,
                bias_vts
            )

            layers.append(layer)

            print(
                f"Layer {layer_index}: "
                f"{n_inputs} -> {n_neurons}"
            )

            previous_size = n_neurons

    print("Network loaded successfully.\n")

    return Network(
        input_size,
        loss_type,
        adam_t,
        layers
    )


# ============================================================
# Prediction
# ============================================================

def predict(network, image):

    x = (
        image.astype(np.float64)
        .reshape(-1)
        / 255.0
    )

    probabilities = network.forward(x)

    predicted = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted]
    )

    return (
        probabilities,
        predicted,
        confidence
    )


def cross_entropy(probabilities, label):

    p = max(
        float(probabilities[label]),
        1e-12
    )

    return -np.log(p)


# ============================================================
# Font
# ============================================================

def get_font(size, bold=False):

    candidates = []

    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]

    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for filename in candidates:

        if Path(filename).exists():

            return ImageFont.truetype(
                filename,
                size
            )

    return ImageFont.load_default()


# ============================================================
# Generate result PNG
# ============================================================

def save_result_png(
    filename,
    image,
    label,
    predicted,
    probabilities,
    loss,
    confidence
):

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    image_scale = 12

    image_width = 28 * image_scale
    image_height = 28 * image_scale

    panel_width = 430

    margin = 20

    width = (
        image_width
        + panel_width
        + margin * 3
    )

    height = max(
        image_height + margin * 2,
        480
    )

    # --------------------------------------------------------
    # Create white canvas
    # --------------------------------------------------------

    canvas = Image.new(
        "RGB",
        (width, height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    # --------------------------------------------------------
    # Fonts
    # --------------------------------------------------------

    font_title = get_font(
        24,
        bold=True
    )

    font_normal = get_font(
        18
    )

    font_small = get_font(
        15
    )

    font_bold = get_font(
        18,
        bold=True
    )

    # --------------------------------------------------------
    # MNIST image
    # --------------------------------------------------------

    image_rgb = np.stack(
        [image, image, image],
        axis=-1
    )

    image_pil = Image.fromarray(
        image_rgb,
        mode="RGB"
    )

    image_pil = image_pil.resize(
        (
            image_width,
            image_height
        ),
        Image.Resampling.NEAREST
    )

    image_x = margin
    image_y = margin

    canvas.paste(
        image_pil,
        (image_x, image_y)
    )

    # Border
    if predicted == label:
        border_color = "green"
    else:
        border_color = "red"

    draw.rectangle(
        [
            image_x - 3,
            image_y - 3,
            image_x + image_width + 2,
            image_y + image_height + 2
        ],
        outline=border_color,
        width=5
    )

    # --------------------------------------------------------
    # Information panel
    # --------------------------------------------------------

    panel_x = (
        image_x
        + image_width
        + margin * 2
    )

    y = margin

    status = (
        "CORRECT"
        if predicted == label
        else "WRONG"
    )

    draw.text(
        (panel_x, y),
        status,
        fill=border_color,
        font=font_title
    )

    y += 45

    draw.text(
        (panel_x, y),
        f"True label:  {label}",
        fill="black",
        font=font_normal
    )

    y += 30

    draw.text(
        (panel_x, y),
        f"Predicted:   {predicted}",
        fill="black",
        font=font_normal
    )

    y += 30

    draw.text(
        (panel_x, y),
        f"Confidence:  {confidence:.12e}",
        fill="black",
        font=font_small
    )

    y += 25

    draw.text(
        (panel_x, y),
        f"True prob.:  {probabilities[label]:.12e}",
        fill="black",
        font=font_small
    )

    y += 25

    draw.text(
        (panel_x, y),
        f"Loss:        {loss:.12f}",
        fill="black",
        font=font_small
    )

    y += 50

    draw.text(
        (panel_x, y),
        "Probabilities",
        fill="black",
        font=font_bold
    )

    y += 35

    # --------------------------------------------------------
    # Probability bars
    # --------------------------------------------------------

    bar_width = 260
    bar_height = 22
    text_width = 35

    for digit in range(NUM_CLASSES):

        probability = float(
            probabilities[digit]
        )

        # Digit
        draw.text(
            (panel_x, y),
            str(digit),
            fill="black",
            font=font_small
        )

        bar_x = (
            panel_x
            + text_width
        )

        # Background
        draw.rectangle(
            [
                bar_x,
                y + 2,
                bar_x + bar_width,
                y + bar_height
            ],
            outline="gray"
        )

        # Bar
        filled_width = int(
            probability * bar_width
        )

        if filled_width > 0:

            if digit == label:
                color = "green"

            elif digit == predicted:
                color = "blue"

            else:
                color = "gray"

            draw.rectangle(
                [
                    bar_x,
                    y + 2,
                    bar_x + filled_width,
                    y + bar_height
                ],
                fill=color
            )

        # Probability
        draw.text(
            (
                bar_x + bar_width + 10,
                y
            ),
            f"{probability:.6e}",
            fill="black",
            font=font_small
        )

        y += 28

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    canvas.save(
        filename,
        format="PNG"
    )


# ============================================================
# Manual mode
# ============================================================

def run_manual(
    network,
    images,
    labels
):

    output_dir = Path(
        OUTPUT_DIRECTORY
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    total = len(images)

    index = 0

    print()
    print("=" * 60)
    print("MANUAL MODE")
    print("=" * 60)
    print()
    print("Enter : next image")
    print("s     : next incorrect prediction")
    print("q     : quit")
    print()
    print(
        f"PNG files will be saved to: "
        f"{output_dir}/"
    )
    print()

    while index < total:

        image = images[index]
        label = int(labels[index])

        probabilities, predicted, confidence = predict(
            network,
            image
        )

        loss = cross_entropy(
            probabilities,
            label
        )

        filename = (
            output_dir
            / f"image_{index:05d}_"
              f"true_{label}_"
              f"pred_{predicted}.png"
        )

        save_result_png(
            filename,
            image,
            label,
            predicted,
            probabilities,
            loss,
            confidence
        )

        # ----------------------------------------------------
        # Terminal information
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print(
            f"Image:                  "
            f"{index + 1}/{total}"
        )

        print(
            f"True label:             "
            f"{label}"
        )

        print(
            f"Predicted label:        "
            f"{predicted}"
        )

        print(
            f"Correct:                "
            f"{'YES' if predicted == label else 'NO'}"
        )

        print(
            f"Prediction confidence:  "
            f"{confidence:.12e}"
        )

        print(
            f"Correct-class probability:"
            f" {probabilities[label]:.12e}"
        )

        print(
            f"Cross-entropy loss:     "
            f"{loss:.12f}"
        )

        print()
        print("Network probabilities:")

        for digit, probability in enumerate(
            probabilities
        ):

            marker = ""

            if digit == predicted:
                marker += " <-- predicted"

            if digit == label:
                marker += " <-- true"

            print(
                f"  {digit}: "
                f"{probability:.12e}"
                f"{marker}"
            )

        print()
        print(f"Saved: {filename}")

        # ----------------------------------------------------
        # Wait for command
        # ----------------------------------------------------

        command = input(
            "\n[Enter] next | [s] next error | [q] quit: "
        ).strip().lower()

        if command == "q":
            break

        elif command == "s":

            index += 1

            while index < total:

                probabilities, predicted, _ = predict(
                    network,
                    images[index]
                )

                label = int(
                    labels[index]
                )

                if predicted != label:
                    break

                index += 1

            continue

        index += 1


# ============================================================
# Summary mode
# ============================================================

def run_summary(
    network,
    images,
    labels
):

    total = len(images)

    correct = 0
    total_loss = 0.0

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    class_correct = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    class_total = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    worst_errors = []
    confident_errors = []

    print("=" * 60)
    print("MNIST NETWORK SUMMARY")
    print("=" * 60)

    for i in range(total):

        probabilities, predicted, confidence = predict(
            network,
            images[i]
        )

        label = int(labels[i])

        loss = cross_entropy(
            probabilities,
            label
        )

        total_loss += loss

        confusion[
            label,
            predicted
        ] += 1

        class_total[label] += 1

        if predicted == label:

            correct += 1
            class_correct[label] += 1

        else:

            worst_errors.append(
                (
                    loss,
                    i,
                    label,
                    predicted,
                    confidence
                )
            )

            confident_errors.append(
                (
                    confidence,
                    i,
                    label,
                    predicted,
                    loss
                )
            )

    accuracy = (
        correct
        / total
        * 100.0
    )

    average_loss = (
        total_loss
        / total
    )

    print()
    print(f"Total images:      {total}")
    print(f"Correct:           {correct}")
    print(f"Incorrect:         {total - correct}")
    print(
        f"Average loss:      "
        f"{average_loss:.12f}"
    )
    print(
        f"Accuracy:          "
        f"{accuracy:.4f}%"
    )

    # --------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PER-CLASS ACCURACY")
    print("=" * 60)

    for digit in range(NUM_CLASSES):

        acc = (
            class_correct[digit]
            / class_total[digit]
            * 100.0
        )

        print(
            f"Digit {digit}: "
            f"{class_correct[digit]:4d}/"
            f"{class_total[digit]:4d} = "
            f"{acc:7.3f}%"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(
        "      "
        + " ".join(
            f"{i:5d}"
            for i in range(10)
        )
    )

    for actual in range(10):

        print(
            f"{actual:3d}: "
            + " ".join(
                f"{confusion[actual, predicted]:5d}"
                for predicted in range(10)
            )
        )

    # --------------------------------------------------------
    # Worst errors
    # --------------------------------------------------------

    worst_errors.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    print()
    print("=" * 60)
    print(
        f"TOP {N_WORST_ERRORS} "
        "WORST ERRORS"
    )
    print("=" * 60)

    for rank, error in enumerate(
        worst_errors[:N_WORST_ERRORS],
        start=1
    ):

        loss, index, label, predicted, confidence = error

        print(
            f"{rank:2d}. "
            f"Image {index:5d}: "
            f"{label} -> {predicted}, "
            f"confidence={confidence:.12e}, "
            f"loss={loss:.12f}"
        )

    # --------------------------------------------------------
    # Most confident wrong predictions
    # --------------------------------------------------------

    confident_errors.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    print()
    print("=" * 60)
    print(
        f"TOP {N_WORST_CONFIDENT_ERRORS} "
        "MOST CONFIDENT WRONG PREDICTIONS"
    )
    print("=" * 60)

    for rank, error in enumerate(
        confident_errors[
            :N_WORST_CONFIDENT_ERRORS
        ],
        start=1
    ):

        confidence, index, label, predicted, loss = error

        print(
            f"{rank:2d}. "
            f"Image {index:5d}: "
            f"{label} -> {predicted}, "
            f"confidence={confidence:.12e}, "
            f"loss={loss:.12f}"
        )

    # --------------------------------------------------------
    # Most common confusion pairs
    # --------------------------------------------------------

    pairs = []

    for actual in range(NUM_CLASSES):

        for predicted in range(NUM_CLASSES):

            if actual == predicted:
                continue

            count = confusion[
                actual,
                predicted
            ]

            if count > 0:

                pairs.append(
                    (
                        count,
                        actual,
                        predicted
                    )
                )

    pairs.sort(
        reverse=True
    )

    print()
    print("=" * 60)
    print("MOST COMMON CONFUSION PAIRS")
    print("=" * 60)

    for count, actual, predicted in pairs[:20]:

        print(
            f"{actual} -> {predicted}: "
            f"{count} errors"
        )

    print()


# ============================================================
# Main
# ============================================================

def main():

    mode = "MANUAL"

    if len(sys.argv) >= 2:
        mode = sys.argv[1].upper()

    if mode not in (
        "MANUAL",
        "SUMMARY"
    ):
        print(
            "Usage:\n"
            "  python MNIST_Display.py MANUAL\n"
            "  python MNIST_Display.py SUMMARY"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Load MNIST
    # --------------------------------------------------------

    images = load_mnist_images(
        MNIST_IMAGES
    )

    labels = load_mnist_labels(
        MNIST_LABELS
    )

    if len(images) != len(labels):
        raise ValueError(
            "Number of images and labels "
            "do not match"
        )

    print(
        f"Loaded {len(images)} "
        f"MNIST test images."
    )

    # --------------------------------------------------------
    # Load network
    # --------------------------------------------------------

    network = load_network(
        NETWORK_FILE
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    if mode == "SUMMARY":

        run_summary(
            network,
            images,
            labels
        )

    else:

        run_manual(
            network,
            images,
            labels
        )


if __name__ == "__main__":
    main()
