from PIL import Image, ImageDraw, ImageDraw, ImageFont
import numpy as np
import textwrap
from typing import List, Literal

def create_image_grid_with_labels(images, nrow, pad_size=10, label_size=20,
                                  font_size=40, background_color=(0, 0, 0),
                                  title=None, title_size=60):
    """
    Creates a grid of images with labels, and optionally adds a title.

    Parameters:
    
    - images: Array of images to be placed in the grid.
    - nrow: Number of rows in the grid.
    - pad_size: Padding size around each image.
    - label_size: Size of the label box (not used in this implementation).
    - font_size: Font size for the labels.
    - background_color: Background color of the grid.
    - title: Optional title for the image grid.
    - title_size: Font size for the title.

    Returns:
    - An image containing the grid with images, labels, and an optional title.
    """
    N, H, W, C = images.shape
    ncol = int(np.ceil(N / nrow))  # Calculate the number of columns needed
    H_pad = H + pad_size * 2
    W_pad = W + pad_size * 2
    title_height = 0

    if title:
        title_height = title_size + pad_size  # Space for title if provided

    grid_height = nrow * H_pad + pad_size + title_height  # Adjust height for title
    grid_width = ncol * W_pad + pad_size

    # Creating an image for the grid
    grid_image = Image.new('RGB', (grid_width, grid_height), color=background_color)

    # Adding the title if provided
    if title:
        draw = ImageDraw.Draw(grid_image)
        font = ImageFont.load_default()  # Adjust the font size for the title
        # Calculate the width and height of the title text
        title_width, title_height = draw.textbbox((0, 0), title, font=font)[2:]
        draw.text(((grid_width - title_width) / 2, pad_size / 2), title,
                  fill=(255, 255, 255), font=font)

    for idx, img_array in enumerate(images):
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(font_size)
        draw.text((5, 5), str(idx), fill=(255, 255, 255), font=font)

        row = idx // ncol
        col = idx % ncol
        start_row = row * H_pad + pad_size + title_height
        start_col = col * W_pad + pad_size
        grid_image.paste(img, (start_col, start_row))

    return grid_image

def print_strings_on_image(string_list,
                           font_size=24,
                           line_spacing=10,
                           image_width=400,
                           max_width=80,
                           numbering=True,
                           numbering_start_idx=0):
    """ 
    Put a list of strings into a PIL image. 
    Optionally, add numbering to the items 
    The point of this is so we can display prompts or retrieval strings with 
    some visual data on the same image. 
    """

    def calculate_image_height(string_list,
                               font_size=font_size,
                               line_spacing=line_spacing,
                               max_width=max_width):
        # Use default font
        font = ImageFont.load_default()

        # Since we're using a default font, adjust line height manually
        line_height = font_size + line_spacing

        # Total height initialization
        total_height = 0

        for index, text in enumerate(string_list, start=numbering_start_idx):
            numbered_text = f"{index}: {text}"
            wrapped_text = textwrap.fill(numbered_text, width=max_width)
            total_lines = wrapped_text.count(
                '\n') + 1  # Counting how many lines will be used
            total_height += total_lines * line_height

        return total_height

    # First, calculate the image height
    image_height = calculate_image_height(string_list, font_size, line_spacing,
                                          max_width)

    # Create a blank image with dynamic height
    image = Image.new('RGB', (image_width, image_height), color='white')
    draw = ImageDraw.Draw(image)

    # Use default font
    font = ImageFont.load_default()

    # Since we're using a default font, adjust line height manually
    line_height = font_size + line_spacing

    # Initial Y position
    y_text = 0

    for index, text in enumerate(string_list, start=numbering_start_idx):
        if numbering:
            # Prepend number before each string
            numbered_text = f"{index}: {text}"
        else:
            numbered_text = text

        # Wrap text, limiting width to 80 characters
        wrapped_text = textwrap.fill(numbered_text, width=max_width)

        # Draw each line
        for line in wrapped_text.split('\n'):
            draw.text((10, y_text), line, fill=(0, 0, 0), font=font)
            y_text += line_height

    return image

def stack_images(img0: Image.Image,
                 img1: Image.Image,
                 mode: Literal['h', 'v'] = 'h',
                 resize_width: bool = False,
                 resize_height: bool = False):
    """ 
    Put a pair of pil images next to each other horizonally if mode='h' or 
    vertically if mode='v'.

    Another method `stack_images_seq` does the same thing but with a sequence 
    of images. 
    """
    # mode should be 'h' or 'v' for stack horizontally or vertically.
    # If resizing by width, adjust both images to the width of the smaller image
    if resize_width:
        min_width = min(img0.width, img1.width)
        img0 = img0.resize(
            (min_width, int(min_width * img0.height / img0.width)))
        img1 = img1.resize(
            (min_width, int(min_width * img1.height / img1.width)))

    # If resizing by height, adjust both images to the height of the smaller image
    if resize_height:
        min_height = min(img0.height, img1.height)
        img0 = img0.resize(
            (int(min_height * img0.width / img0.height), min_height))
        img1 = img1.resize(
            (int(min_height * img1.width / img1.height), min_height))

    # Stack images horizontally
    if mode == 'h':
        max_height = max(img0.height, img1.height)
        total_width = img0.width + img1.width
        dst = Image.new('RGB', (total_width, max_height))
        dst.paste(img0, (0, 0))
        dst.paste(img1, (img0.width, 0))

    # Stack images vertically
    elif mode == 'v':
        total_height = img0.height + img1.height
        max_width = max(img0.width, img1.width)
        dst = Image.new('RGB', (max_width, total_height))
        dst.paste(img0, (0, 0))
        dst.paste(img1, (0, img0.height))

    return dst

def add_border_to_img(img, border_position='top', border_size=10):
    """
    Add a black border to an image on the specified side using the stack_images function.

    :param img: The original PIL Image object.
    :param border_size: The thickness of the border in pixels.
    :param border_position: The position of the border ('top', 'bottom', 'left', 'right').
    :return: A new PIL Image object with the border added.
    """
    # Create a black border image based on the specified position and size
    if border_position in ['top', 'bottom']:
        border_img = Image.new('RGB', (img.width, border_size),
                               color=(0, 0, 0))
    else:  # 'left' or 'right'
        border_img = Image.new('RGB', (border_size, img.height),
                               color=(0, 0, 0))

    # Use the stack_images function to add the border to the original image
    if border_position == 'top':
        return stack_images(border_img, img, mode='v', resize_width=True)
    elif border_position == 'bottom':
        return stack_images(img, border_img, mode='v', resize_width=True)
    elif border_position == 'left':
        return stack_images(border_img, img, mode='h', resize_height=True)
    else:  # 'right'
        return stack_images(img, border_img, mode='h', resize_height=True)

def stack_images_seq(imgs: List[Image.Image],
                     mode: Literal['h', 'v'] = 'h',
                     resize_width: bool = False,
                     resize_height: bool = False):
    """
    Iteratively call stack_images(img0,img1) over a sequence of images
    """
    if len(imgs) == 1:
        return imgs[0]

    img_stacked = stack_images(imgs[0],
                               imgs[1],
                               mode=mode,
                               resize_width=resize_width,
                               resize_height=resize_height)
    for img_right in imgs[2:]:
        img_stacked = stack_images(img_stacked,
                                   img_right,
                                   mode=mode,
                                   resize_width=resize_width,
                                   resize_height=resize_height)

    return img_stacked

