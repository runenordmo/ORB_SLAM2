from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import ImageDraw ,Image
import ctypes as ct
import os

image_width = 1241;
image_height = 376;


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



def get_descriptors(nr, features):
    assert features.shape == (1,56,56,256), "features must be 1x56x56x256"

    descriptors = []
    for x in range(1, 54):
        for y in range(1, 54):
            cut = features[0,y,x,:] 
            # if np.sum(cut) > 80000:
            #     img_x = nr*56 + x
            #     s = np.sum(cut)
            #     # print(s)
            #     print(f"({img_x},{y}) = {s}")
            #     descriptors.append((img_x,y,cut))

            for d in range(0,255):
                if features[0,y,x,d] > 5000:
                    img_x = nr*56 + x
                    descriptors.append((img_x,y,cut))
                    break

    return descriptors


def save_debug_img(img_name, parts, descriptors):
    new_img = Image.new('RGB', (224*5,224))
    for i in range(0,len(parts)):
        new_img.paste(parts[i], (224*i,0))

    draw = ImageDraw.Draw(new_img)
    for desc in descriptors:
        x,y,arr = desc
        draw.rectangle((4*x,4*y,4*(x+1),4*(y+1)), outline=(255,0,0))

    new_img.save('output/' + img_name + '.with_features.png')


def write_descriptors(file, descriptors):
    for desc in descriptors:
        x,y,arr = desc
        file.write(ct.c_int32(x))
        file.write(ct.c_int32(y))
        arr.astype('float32').tofile(file)
    file.write(ct.c_int32(-1))
    file.write(ct.c_int32(-1))
    



base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_conv3').output)


path = 'data/00/'
images = get_images(path, '.png')
file = open('output/descriptors.dat', 'wb')
for img_name in images:
    img = image.load_img(path + img_name)
    parts = split_image(img)
    descriptors = []
    for i in range(0,len(parts)):
        features = produce_featuremaps(model, parts[i])
        descriptors += get_descriptors(i, features)

    print(f'{img_name}: Found {len(descriptors)} descriptors')
    save_debug_img(img_name, parts, descriptors)
    write_descriptors(file, descriptors)


