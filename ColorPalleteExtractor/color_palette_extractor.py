import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

def extract_colors(image_path, num_colors=5):
    # Load image
    image = Image.open(image_path)
    image = image.resize((100, 100))  # Resize for faster processing
    pixels = np.array(image).reshape(-1, 3)  # Flatten image to pixel array

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)  # Extract dominant colors

    return colors

def plot_palette(colors):
    """Display the extracted color palette."""
    plt.figure(figsize=(8, 2))
    plt.axis("off")
    plt.imshow([colors], aspect="auto")
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = r"C:\Coding\Python Projects\ColorPaletteExtractor\my_image.jpg"
    colors = extract_colors(image_path, num_colors=5)
    plot_palette(colors)

