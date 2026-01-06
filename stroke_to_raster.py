# Convert vector stroke data to raster images
# By Dan Jackson, 2026

# pip install rdp cairocffi

import numpy as np
from rdp import rdp

# Hack to add Homebrew's lib path to ctypes search paths on macOS
#import sys
#if sys.platform == "darwin":
#    from ctypes.macholib import dyld
#    dyld.DEFAULT_LIBRARY_FALLBACK.append("/opt/homebrew/lib")

import cairocffi as cairo


def stroke_to_raster(vector_strokes):
    # Normalize and simplify strokes
    simplified_strokes = _normalize_and_simplify_strokes(vector_strokes)

    # Convert to raster image
    side = 28
    raster_images = _vector_to_raster([simplified_strokes], side)
    raster_image = raster_images[0]

    # Convert to one array per row
    output_image = np.zeros((side, side), dtype=np.uint8)
    for y in range(side):
        for x in range(side):
            output_image[y, x] = raster_image[y * side + x]

    return output_image


# Based on: https://github.com/googlecreativelab/quickdraw-dataset?tab=readme-ov-file#simplified-drawing-files-
def _normalize_and_simplify_strokes(vector_strokes, new_max = 255.0, epsilon = 2.0):
    # vector_strokes is a Python list of strokes, each stroke is a list of points, each point is a two-element [x,y] list

    # Return early if no strokes
    if len(vector_strokes) == 0:
        return vector_strokes

    # Step 1. Align the drawing to the top-left corner, to have minimum values of 0.
    min_x = min([min([point[0] for point in stroke]) for stroke in vector_strokes])
    min_y = min([min([point[1] for point in stroke]) for stroke in vector_strokes])
    for stroke in vector_strokes:
        for point in stroke:
            point[0] -= min_x
            point[1] -= min_y

    # Step 2. Uniformly scale the drawing, to have a maximum value of 255.
    max_x = max([max([point[0] for point in stroke]) for stroke in vector_strokes])
    max_y = max([max([point[1] for point in stroke]) for stroke in vector_strokes])
    max_value = max(max_x, max_y)
    if max_value == 0:
        max_value = 1.0
    scale = new_max / max_value
    for stroke in vector_strokes:
        for point in stroke:
            point[0] *= scale
            point[1] *= scale
    
    # Step 3. Resample all strokes with a 1 pixel spacing.
    # (assume this means to convert to integer pixel coordinates???)
    for stroke in vector_strokes:
        for point in stroke:
            point[0] = int(round(point[0]))
            point[1] = int(round(point[1]))
    
    # Step 4. Simplify all strokes using the Ramer–Douglas–Peucker algorithm with an epsilon value of 2.0.
    simplified = []
    for stroke in vector_strokes:
        simplified_points = rdp(stroke, epsilon=epsilon)
        simplified.append(simplified_points)

    # Transpose coordinates as expected for vector_to_raster
    for stroke_idx in range(len(simplified)):
        stroke = simplified[stroke_idx]
        xs = np.array([point[0] for point in stroke])
        ys = np.array([point[1] for point in stroke])
        simplified[stroke_idx] = (xs, ys)

    return simplified


# This function from: https://github.com/googlecreativelab/quickdraw-dataset/issues/19#issuecomment-402247262
def _vector_to_raster(vector_images, side=28, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
    """
    padding and line_diameter are relative to the original 256x256 image.
    """
    
    original_side = 256.
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()
        
        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1,1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)        
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)
    
    return raster_images
