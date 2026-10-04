"""Convert the original MNIST training IDX files to label-first integer CSV."""

import argparse
import csv
import gzip
import hashlib
import struct
from pathlib import Path


def open_idx(path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def convert(images_path, labels_path, output):
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open_idx(images_path) as images, open_idx(labels_path) as labels:
        image_header = images.read(16)
        label_header = labels.read(8)
        if len(image_header) != 16 or len(label_header) != 8:
            raise ValueError("Truncated MNIST IDX header")
        image_magic, n_images, rows, columns = struct.unpack(">IIII", image_header)
        label_magic, n_labels = struct.unpack(">II", label_header)
        if (image_magic, label_magic) != (2051, 2049) or (rows, columns) != (28, 28):
            raise ValueError("Expected MNIST unsigned-byte 28x28 images and labels")
        if n_images != n_labels or n_images != 60000:
            raise ValueError("Expected 60,000 matched MNIST training images and labels")
        with output.open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            for _ in range(n_images):
                label = labels.read(1)
                pixels = images.read(784)
                if len(label) != 1 or len(pixels) != 784:
                    raise ValueError("Truncated MNIST IDX payload")
                writer.writerow((label[0], *pixels))
        if images.read(1) or labels.read(1):
            raise ValueError("MNIST IDX files contain trailing data")
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-sha256")
    args = parser.parse_args()
    actual = convert(args.images, args.labels, args.output)
    print(f"Wrote {args.output}; SHA-256 {actual}")
    if args.expect_sha256 and actual.lower() != args.expect_sha256.lower():
        raise SystemExit("CSV digest differs from the published experiment input")


if __name__ == "__main__":
    main()
