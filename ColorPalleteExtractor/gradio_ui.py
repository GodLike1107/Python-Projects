import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import gradio as gr
from io import BytesIO
import os


def extract_colors(image, num_colors=4):
    """Extracts the dominant colors from an image using K-Means clustering."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = image.reshape((-1, 3))  # Flatten the image into a 2D array
    kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=10)
    kmeans.fit(image)
    palette = kmeans.cluster_centers_.astype(int)  # Convert to integer RGB values
    hex_codes = ['#{:02x}{:02x}{:02x}'.format(*color) for color in palette]
    return palette, hex_codes


def plot_palette(palette, file_format):
    """Creates a color palette image with hex codes and saves in the selected format."""
    num_colors = len(palette)
    fig, ax = plt.subplots(1, figsize=(num_colors * 1.5, 2.5))  # Increase figure height
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow([palette], aspect='auto')

    # Add hex codes to the palette
    for i, color in enumerate(palette):
        hex_code = '#{:02x}{:02x}{:02x}'.format(*color)
        ax.text(i / num_colors + 0.02, 0.5, hex_code, fontsize=12, ha='left', va='center', transform=ax.transAxes,
                color='white', bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.3'))

    buf = BytesIO()
    plt.savefig(buf, format=file_format, bbox_inches='tight', pad_inches=0.8)  # Increase padding
    plt.close(fig)
    buf.seek(0)

    output_path = f"palette.{file_format}"
    with open(output_path, "wb") as f:
        f.write(buf.getbuffer())

    return output_path


def process_image(image_path, num_colors, file_format):
    """Processes the uploaded image, extracts colors, and generates a color palette."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image file.")

    palette, hex_codes = extract_colors(image, num_colors)
    palette_img_path = plot_palette(palette, file_format)
    return palette_img_path, '\n'.join(hex_codes)


with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Image Color Palette Extractor")
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload Image")
    num_colors_slider = gr.Slider(minimum=3, maximum=10, value=4, step=1, label="Number of Colors")
    file_format_radio = gr.Radio(choices=["png", "jpg"], value="png", label="Palette Image Format")
    extract_button = gr.Button("Extract Colors")
    palette_output = gr.Image(type="filepath", label="Color Palette")
    hex_output = gr.Textbox(label="Extracted Hex Codes", interactive=False)
    extract_button.click(process_image, inputs=[image_input, num_colors_slider, file_format_radio],
                         outputs=[palette_output, hex_output])

demo.launch()
