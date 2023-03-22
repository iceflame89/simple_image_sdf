
from PIL import Image, ImageFont, ImageDraw
import scipy.ndimage as nd
import numpy as np
import cv2

PIXELS = 2 ** 22

def save_2d_texture(texture, sdf_vis_path):
    a = np.abs(texture)
    lo, hi = a.min(), a.max()
    a = (a - lo) / (hi - lo) * 255
    im = Image.fromarray(a.astype('uint8'))
    im.save(sdf_vis_path)

def resize(np2d, width):
    h,w = np2d.shape    
    if w != width:
        height = int(h* width/w + 0.5)
        np2d = cv2.resize(np2d, (width, height))
    return np2d

def texture_to_img(texture, outpath, width=100):
    texture = resize(texture, width=width)

    a = np.zeros(texture.shape)
    a[texture>0] = 0
    a[texture<=0] = 255
    im = Image.fromarray(a.astype('uint8'))
    im.save(outpath)


def _load_image(thing):
    if isinstance(thing, str):
        return Image.open(thing)
    elif isinstance(thing, (np.ndarray, np.generic)):
        return Image.fromarray(thing)
    return Image.fromarray(np.array(thing))


def _sdf(width, height, pixels, px, py, im):
    tw, th = im.size

    # downscale image if necessary
    factor = (pixels / (tw * th)) ** 0.5
    if factor < 1:
        tw, th = int(round(tw * factor)), int(round(th * factor))
        px, py = int(round(px * factor)), int(round(py * factor))
        im = im.resize((tw, th))
    img = np.array(im)

    # convert to numpy array and apply distance transform
    im = im.convert('1')
    a = np.array(im)
    inside = -nd.distance_transform_edt(a)
    outside = nd.distance_transform_edt(~a)
    texture = np.zeros(a.shape)
    texture[a] = inside[a]
    texture[~a] = outside[~a]

    # save debug image
    texture = resize(texture, width=64)
    save_2d_texture(texture, 'bf_sdf.png')
    texture_to_img(texture, 'bf_rec_1000.png', width=500)
    
    resize_img = resize(img, width=128)
    im = Image.fromarray(resize_img.astype('uint8'))
    im.save('bf_resize128.png')
    resize_img = resize(resize_img, width=1000)
    im = Image.fromarray(resize_img.astype('uint8'))
    im.save('bf_resize128_1000.png')

def image(thing, width=None, height=None, pixels=PIXELS):
    im = _load_image(thing).convert('L')
    return _sdf(width, height, pixels, 0, 0, im)

def mask2sdf(a):
    inside = -nd.distance_transform_edt(a)
    outside = nd.distance_transform_edt(~a)
    texture = np.zeros(a.shape)
    texture[a] = inside[a]
    texture[~a] = outside[~a]
    return texture



path='examples/butterfly.png'
sdf = image(path)


