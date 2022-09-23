import cv2
import numpy as np
import requests


def load_image(url: str) -> np.ndarray:
    """Load an image from a URL."""
    resp = requests.get(url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    im = cv2.imdecode(image, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def split_tiles(image: np.ndarray, nx: int, ny: int) -> list:
    """Split the image into tiles."""
    width = image.shape[0] // ny
    height = image.shape[1] // nx
    return [image[x:x + width, y:y + height] for x in range(0, image.shape[0], width) for y in
            range(0, image.shape[1], height)]
