#!/usr/bin/env python3

from features import Image, FeatureDetector
import ctypes as ct
import os
import argparse


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


def save_debug_img(output, img_name, img, descriptors):
    new_img = img.draw_descriptors(descriptors)
    new_img.save(output + img_name + '.with_features.png')


def write_descriptors(file, descriptors):
    for desc in descriptors:
        x,y = desc.kp()
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
parser.add_argument("-t", "--threshold", help="feature threshold", type=int, default=500)
parser.add_argument("-n", "--number", help="number of keypoints per region", type=int, default=1000)
parser.add_argument("-d", "--debug", help="saves a debug image", action="store_true")
args = parser.parse_args()

assert args.path.endswith('/'), "path must end with '/'"
assert args.output.endswith('/'), "output must end with '/'"
assert args.ext.startswith('.'), "ext must start with '.'"


detector = FeatureDetector(args.threshold, args.number)

images = get_images(args.path, args.ext)
file = open(args.output + args.file, 'wb')
for img_name in images:
    img = Image(args.path + img_name)
    descriptors = detector.detect(img)

    print(f'{img_name}: Found {len(descriptors)} descriptors')
    if args.debug:
        save_debug_img(args.output, img_name, img, descriptors)
    write_descriptors(file, descriptors)

