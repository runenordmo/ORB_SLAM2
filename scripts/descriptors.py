from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from model import Model
import numpy as np
import math
from PIL import ImageDraw, ImageOps
import ctypes as ct
import os
import argparse

CNN_INP_SZ = 224
CNN_FM_SZ = 28
CNN_FM_CNT = 512
CNN_SCALE = CNN_INP_SZ / CNN_FM_SZ



class Image:
    def __init__(self, name, img):
        self.name = name
        self.width = img.width
        self.height = img.height
        self._orig = img

        cols = math.ceil(img.width / CNN_INP_SZ)
        rows = math.ceil(img.height / CNN_INP_SZ)
        xpad = cols * CNN_INP_SZ - img.width
        ypad = rows * CNN_INP_SZ - img.height
        padded_img = ImageOps.expand(img, (0,0,xpad,ypad))
        assert padded_img.size == (cols * CNN_INP_SZ, rows * CNN_INP_SZ), "Padded image size is incorrect"

        self.parts = []
        for r in range(rows):
            for c in range(cols):
                x = c * CNN_INP_SZ
                y = r * CNN_INP_SZ
                assert x <= padded_img.width - CNN_INP_SZ, "x is out of range"
                assert y <= padded_img.height - CNN_INP_SZ, "y is out of range"
                subimage = padded_img.crop(box=(x,y,x+CNN_INP_SZ,y+CNN_INP_SZ))
                self.parts.append((x, y, subimage))
        assert len(self.parts) == rows * cols, "Number of image parts is incorrect"

    def draw_descriptors(self, descriptors):
        copy = self._orig.copy()

        draw = ImageDraw.Draw(copy)
        for desc in descriptors:
            x,y = desc.kp
            draw.rectangle((x-4,y-4,x+4,y+4), outline=desc.color())

        return copy


class KpDesc:
    def __init__(self, kp, desc, fm):
        self.kp = kp
        self.desc = desc
        self._fm = fm

    def color(self):
        fm = self._fm
        mask = 7
        col_step = 18
        r = (fm & (mask << 6)) >> 6
        g = (fm & (mask << 3)) >> 3
        b = (fm & (mask << 0)) >> 0
        r = 255 - (r * col_step)
        g = 255 - (g * col_step)
        b = 255 - (b * col_step)
        return (r,g,b)



def get_images(path, ext):
    assert path.endswith('/'), "path must end with '/'"
    assert ext.startswith('.'), "ext must start with '.'"
    files = os.listdir(path)
    files.sort()
    images = []
    for f in files:
        if f.endswith(ext):
            images.append(f)
    return images


def produce_featuremaps(model, img):
    assert img.size == (CNN_INP_SZ, CNN_INP_SZ), "Image size does not match CNN"

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    featuremaps = model.predict(x)
    return np.squeeze(featuremaps, axis=0)


def find_keypoints(featuremaps, threshold):
    assert featuremaps.shape == (CNN_FM_SZ, CNN_FM_SZ, CNN_FM_CNT), "Unexpected feature maps shape"

    keypoints = []
    for i,fm in enumerate(np.rollaxis(featuremaps,2)):
        max = np.unravel_index(np.argmax(fm, axis=None), fm.shape)
        if fm[max] > threshold:
            x = max[1]
            y = max[0]
            keypoints.append((x,y,i))

    return keypoints


def detect_features(model, img, threshold, border):
    descriptors = []
    for x_img,y_img,img_part in img.parts:
        featuremaps = produce_featuremaps(model, img_part)

        keypoints = find_keypoints(featuremaps, threshold)
        for x_kp,y_kp,fm in keypoints:
            x = int(x_img + x_kp*CNN_SCALE + CNN_SCALE/2)
            y = int(y_img + y_kp*CNN_SCALE + CNN_SCALE/2)
            assert x < x_img + CNN_INP_SZ, "x is out of range"
            assert y < y_img + CNN_INP_SZ, "y is out of range"
            if border < x < img.width-border and border < y < img.height:
                desc = featuremaps[y_kp,x_kp,:]
                descriptors.append(KpDesc((x,y),desc,fm))

    return descriptors


def save_debug_img(output, img, descriptors):
    new_img = img.draw_descriptors(descriptors)
    new_img.save(output + img.name + '.with_features.png')


def write_descriptors(file, descriptors):
    for desc in descriptors:
        x,y = desc.kp
        arr = desc.desc
        file.write(ct.c_int32(x))
        file.write(ct.c_int32(y))
        arr.astype('float32').tofile(file)
    file.write(ct.c_int32(-1))
    file.write(ct.c_int32(-1))



parser = argparse.ArgumentParser()
parser.add_argument("path", help="path to image sequence")
parser.add_argument("-o", "--output", help="path to store output files", default="output/")
parser.add_argument("-f", "--file", help="name of descriptors file", default="descriptors.dat")
parser.add_argument("-e", "--ext", help="extension of the image files", default=".png")
parser.add_argument("-t", "--threshold", help="feature threshold", type=int, default=1000)
parser.add_argument("-b", "--border", help="width of insensitive border where features are ignored", type=int, default=2*CNN_SCALE)
parser.add_argument("-d", "--debug", help="saves a debug image", action="store_true")
args = parser.parse_args()

assert args.path.endswith('/'), "path must end with '/'"
assert args.output.endswith('/'), "output must end with '/'"
assert args.ext.startswith('.'), "ext must start with '.'"


base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_conv3').output)


images = get_images(args.path, args.ext)
file = open(args.output + args.file, 'wb')
for img_name in images:
    img = image.load_img(args.path + img_name)
    tmp = Image(img_name, img)
    descriptors = detect_features(model, tmp, args.threshold, args.border)

    print(f'{img_name}: Found {len(descriptors)} descriptors')
    if args.debug:
        save_debug_img(args.output, tmp, descriptors)
    write_descriptors(file, descriptors)

