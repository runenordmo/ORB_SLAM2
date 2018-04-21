from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from model import Model
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import ImageDraw ,Image
import ctypes as ct
import os
import argparse

image_width = 1241;
image_height = 376;

depth = 512
fm_size = 28

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


def split_image(img):
    assert img.size == (image_width, image_height), "Image size does not match"

    htimes = math.floor(img.width/224);
    vtimes = math.floor(img.height/224);

    output = [];

    for i in range(0, htimes):
        new_width = img.width/htimes
        new_height = img.height/vtimes
        img_part = img.crop(box=(i*new_width, 0, (i+1)*new_width, new_height))
        output.append(img_part.resize((224,224)))

    return output



def produce_featuremaps(model, img):
    assert img.size == (224, 224), "Image size must be 224x224"

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return model.predict(x)

def calc_color(d):
    assert d < 512, "Only color mapping available until 512"

    mask = 7
    r = (d & (mask << 6)) >> 6
    g = (d & (mask << 3)) >> 3
    b = (d & (mask << 0)) >> 0
    r = 255 - (r * 18)
    g = 255 - (g * 18)
    b = 255 - (b * 18)
    return (r,g,b)

def get_descriptors(nr, features):
    assert features.shape == (1,fm_size,fm_size,depth), "features must be 1x56x56x256"

    descriptors = []
    for d in range(0,depth-1):
        tmp = features[0,:,:,d]
        max = np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)
        # print(str(max) + ': ' + str(tmp[max]))
        if tmp[max] > 999:
            x = max[1]
            img_x = nr*fm_size + x
            y = max[0]
            cut = features[0,y,x,:] 
            descriptors.append((img_x,y,cut,calc_color(d)))

    # for x in range(1, 54):
    #     for y in range(1, 54):
    #         cut = features[0,y,x,:] 
    #         # if np.sum(cut) > 80000:
    #         #     img_x = nr*56 + x
    #         #     s = np.sum(cut)
    #         #     # print(s)
    #         #     print(f"({img_x},{y}) = {s}")
    #         #     descriptors.append((img_x,y,cut))

    #         for d in range(0,255):
    #             if features[0,y,x,d] > 5000:
    #                 img_x = nr*56 + x
    #                 descriptors.append((img_x,y,cut))
    #                 break

    return descriptors


def save_debug_img(output, img_name, parts, descriptors):
    new_img = Image.new('RGB', (224*5,224))
    for i in range(0,len(parts)):
        new_img.paste(parts[i], (224*i,0))

    draw = ImageDraw.Draw(new_img)
    for desc in descriptors:
        x,y,arr,col = desc
        draw.rectangle((8*x,8*y,8*(x+1),8*(y+1)), outline=col)

    new_img.save(output + img_name + '.with_features.png')


def write_descriptors(file, descriptors):
    for desc in descriptors:
        x,y,arr,_ = desc
        file.write(ct.c_int32(x))
        file.write(ct.c_int32(y))
        arr.astype('float32').tofile(file)
    file.write(ct.c_int32(-1))
    file.write(ct.c_int32(-1))
    

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="path to image sequence", default="data/00/")
parser.add_argument("-o", "--output", help="path to store output files", default="output/")
parser.add_argument("-e", "--ext", help="extension of the image files", default=".png")
args = parser.parse_args()

assert args.path.endswith('/'), "path must end with '/'"
assert args.output.endswith('/'), "output must end with '/'"
assert args.ext.startswith('.'), "ext must start with '.'"


base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_conv3').output)


images = get_images(args.path, args.ext)
file = open(args.output + 'descriptors.dat', 'wb')
for img_name in images:
    img = image.load_img(args.path + img_name)
    parts = split_image(img)
    descriptors = []
    for i in range(0,len(parts)):
        features = produce_featuremaps(model, parts[i])
        descriptors += get_descriptors(i, features)

    print(f'{img_name}: Found {len(descriptors)} descriptors')
    save_debug_img(args.output, img_name, parts, descriptors)
    write_descriptors(file, descriptors)


