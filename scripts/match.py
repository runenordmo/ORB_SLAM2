#!/usr/bin/env python3

from features import Image, FeatureDetector, CNN, refine_max, KpDesc
from PIL import ImageDraw, Image as PILImage
from pathlib import Path
import numpy as np
import scipy.misc
import argparse
import os


def match_internal(descriptors1, descriptors2, swap = False):
    matches = []
    for i,desc1 in enumerate(descriptors1):
        argmin = None
        dist_min = np.inf
        for j,desc2 in enumerate(descriptors2):
           dist = np.linalg.norm(desc1.desc - desc2.desc)
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
            x1,y1 = descriptors1[i].kp()
            x2,y2 = descriptors2[j].kp()
            draw.line((x1,y1, x2,y2+img1.height), fill=(0,255,0))

    new_img.save(output + 'matches.png')


def create_fm_images(detector, img, output, number):
    fms = detector.create_featuremaps(img)

    images = []
    for i,fm in enumerate(np.rollaxis(fms, 2)):
        name = output + str(i) + '.tmp.png'
        scipy.misc.imsave(name, scipy.misc.imresize(fm, size=int(CNN.SCALE*100)))
        images.append(PILImage.open(name).crop(box=(0,0,img.width,img.height)).convert('RGB'))
        try:
            os.remove(name)
        except OSError as e:
            print("Error: %s - %s." % (e.filename,e.strerror))

    regions = detector.create_regions(fms, img.size)
    for region in regions:
        for kp in region.get_top_keypoints(number):
            x,y = refine_max(kp.x, kp.y, fms[:,:,kp.fm])
            x = int(x*CNN.SCALE + CNN.SCALE/2)
            y = int(y*CNN.SCALE + CNN.SCALE/2)
            if img.inside(x, y):
                draw = ImageDraw.Draw(images[kp.fm])
                d = KpDesc((x,y),None,kp.fm)
                draw.ellipse((x-2,y-2,x+2,y+2), outline=d.color())
    
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="images to match", metavar="IMG", nargs=2)
    parser.add_argument("-o", "--output", help="path to store output file", default="output/")
    parser.add_argument("-t", "--threshold", help="feature threshold", type=int, default=500)
    parser.add_argument("-n", "--number", help="number of keypoints per region", type=int, default=3)
    parser.add_argument("-l", "--lines", help="suppress drawing match lines", action="store_false")
    args = parser.parse_args()
    assert args.output.endswith('/'), "output must end with '/'"

    detector = FeatureDetector(args.threshold, args.number)
    img1 = Image(args.images[0])
    img2 = Image(args.images[1])

    # create matches.png
    descriptors1 = detector.detect(img1)
    print(f'Image 1: Found {len(descriptors1)} descriptors')

    descriptors2 = detector.detect(img2)
    print(f'Image 2: Found {len(descriptors2)} descriptors')

    matches = match(descriptors1,descriptors2)
    print(f'Found {len(matches)} matches')
    draw_matches(args.output, img1, img2, descriptors1, descriptors2, matches, args.lines)

    # create feature maps images
    images1 = create_fm_images(detector, img1, args.output, args.number)
    images2 = create_fm_images(detector, img2, args.output, args.number)

    fm_images = []
    assert len(images1) == len(images2), 'Image arrays must be same size'
    for i in range(len(images1)):
        assert images1[i].size == images2[i].size, "Image sizes are different"
        new_img = PILImage.new('RGB', (images1[i].width, 2*images1[i].height))
        new_img.paste(images1[i], (0,0))
        new_img.paste(images2[i], (0,images1[i].height))
        fm_images.append(new_img)

    unexpected = 0
    for i,j in matches:
        if descriptors1[i]._fm == descriptors2[j]._fm:
            color = (0,255,0)
        else:
            color = (255,0,0)
            unexpected = unexpected + 1
            draw = ImageDraw.Draw(fm_images[descriptors2[j]._fm])
            x1,y1 = descriptors1[i].kp()
            x2,y2 = descriptors2[j].kp()
            draw.line((x1,y1, x2,y2+img1.height), fill=color)

        draw = ImageDraw.Draw(fm_images[descriptors1[i]._fm])
        x1,y1 = descriptors1[i].kp()
        x2,y2 = descriptors2[j].kp()
        draw.line((x1,y1, x2,y2+img1.height), fill=color)
    print(f'Unexpected {unexpected} matches')

    for i,image in enumerate(fm_images):
        name = args.output + str(i) + '.png'
        image.save(name)


if __name__ == "__main__":
    main()

