from PIL import Image


def image_grid(imgs, rows=None, cols=None, size=None, reordering=False):
    """Takes a list of PIL images and forms them into a grid as a single PIL image"""
    num_images = len(imgs)

    if rows is not None:
        cols = (num_images + rows - 1) // rows
    elif cols is not None:
        rows = (num_images + cols - 1) // cols
    else:
        raise ValueError("Need to specify rows or cols")

    if size is None:
        w, h = imgs[0].size
    else:
        w, h = size

    if reordering:
        cols, rows = rows, cols

    grid_width = cols * w
    grid_height = rows * h

    # print(reordering, cols, rows, grid_width, grid_height)

    # true_size = (grid_width, grid_height) if not reordering else (grid_height, grid_width)
    grid = Image.new(
        "RGB",
        size=(grid_height, grid_width) if reordering else (grid_width, grid_height),
        color=(255, 255, 255),
    )

    for i, img in enumerate(imgs):
        if size is not None:
            img = img.resize(size)
        grid.paste(
            img,
            box=(i % cols * w, i // cols * h) if not reordering else (i // cols * h, i % cols * w),
        )

    return grid
