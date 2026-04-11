import numpy as np
import os
import matplotlib.pyplot as plt
from imageio import imread


def percentage_saturated_pixels(image: np.ndarray) -> float:
    """
    Computes the percentage of saturated pixels in an image.
    Saturated = pixel value == max possible value for the dtype.
    """
    # Handle multi-channel images (RGB) by checking per pixel
    if image.ndim == 3:
        # A pixel is considered saturated if ANY channel is saturated
        max_val = np.iinfo(image.dtype).max
        saturated = np.any(image == max_val, axis=-1)
    else:
        max_val = np.iinfo(image.dtype).max
        saturated = image == max_val

    return 100.0 * np.count_nonzero(saturated) / saturated.size


if __name__ == "__main__":
    list_images = []
    folder_path = "images/SNR image bank/JAXA"

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".tiff", ".png")):
            image_path = os.path.join(folder_path, filename)
            list_images.append(image_path)

    list_images.sort()  # keeps plot ordering consistent

    saturated_percentages = []

    for image_path in list_images:
        img = imread(image_path)
        pct_sat = percentage_saturated_pixels(img)
        saturated_percentages.append(pct_sat)

        print(f"{os.path.basename(image_path)}: {pct_sat:.3f}% saturated")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(saturated_percentages)),
             saturated_percentages,
             marker="o")
    plt.plot(2,
         saturated_percentages[2],
         marker="o",
         color="red")
    plt.xlabel("Image index")
    plt.ylabel("Percentage of saturated pixels (%)")
    plt.title("Saturated Pixel Percentage per Image")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
