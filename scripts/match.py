#!/usr/bin/env python3

from features import *
from PIL import ImageDraw, Image as PILImage
from pathlib import Path
import argparse


def get_image_pair(path):
    img1_path = Path(path)
    
    img1_name = img1_path.stem
    ext = img1_path.suffix
    assert img1_name.isnumeric(), "img1_name must be numeric"

    img2_name = '{:06d}'.format(int(img1_name) + 1) + ext
    img1 = Image(img1_path)
    img2 = Image(img1_path.with_name(img2_name))
    return (img1, img2)


def match_internal(descriptors1, descriptors2, swap = False):
    matches = []
    for i,desc1 in enumerate(descriptors1):
        argmin = None
        dist_min = np.inf
        for j,desc2 in enumerate(descriptors2):
           dist = np.linalg.norm(desc1.desc - desc2.desc); 
           if dist < dist_min:
               dist_min = dist
               argmin = j

        assert argmin >= 0, "no neighbour found"
        if swap:
            matches.append((argmin,i))
        else:
            matches.append((i,argmin))
    return matches

def match(descriptors1, descriptors2):
    matches1 = match_internal(descriptors1, descriptors2)
    matches2 = match_internal(descriptors2, descriptors1, True)
    sa = set(matches1)
    sb = set(matches2)
    c = sa.intersection(sb)
    return c
    

def draw_matches(output, img1, img2, descriptors1, descriptors2, matches, drawLines):
    assert img1.size == img2.size, "Image sizes are different"
    new_img = PILImage.new('RGB', (img1.width, 2*img1.height))
    new_img.paste(img1.draw_descriptors(descriptors1), (0,0))
    new_img.paste(img2.draw_descriptors(descriptors2), (0,img1.height))

    draw = ImageDraw.Draw(new_img)
    if drawLines:
        for i,j in matches:
            x1,y1 = descriptors1[i].kp
            x2,y2 = descriptors2[j].kp
            draw.line((x1,y1, x2,y2+img1.height), fill=(0,255,0))

    new_img.save(output + 'matches.png')



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="path to image (in sequence)")
parser.add_argument("-o", "--output", help="path to store output file", default="output/")
parser.add_argument("-t", "--threshold", help="feature threshold", type=int, default=1000)
parser.add_argument("-b", "--border", help="width of insensitive border where features are ignored", type=int, default=16)
parser.add_argument("-l", "--lines", help="suppress drawing match lines", action="store_false")
args = parser.parse_args()

assert args.output.endswith('/'), "output must end with '/'"
assert args.image != None, "image is mandatory"


detector = FeatureDetector(args.threshold, args.border)

img1, img2 = get_image_pair(args.image)

descriptors1 = detector.detect(img1)
print(f'Image 1: Found {len(descriptors1)} descriptors')

descriptors2 = detector.detect(img2)
print(f'Image 2: Found {len(descriptors2)} descriptors')

matches = match(descriptors1,descriptors2)
print(f'Found {len(matches)} matches')
draw_matches(args.output, img1, img2, descriptors1, descriptors2, matches, args.lines)

