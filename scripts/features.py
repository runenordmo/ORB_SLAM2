from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from model import Model
import numpy as np
import math
from PIL import ImageDraw, ImageOps


CNN_INP_SZ = 224
CNN_FM_SZ = 28
CNN_FM_CNT = 512
CNN_SCALE = CNN_INP_SZ / CNN_FM_SZ

REGION_CNT = 7
REGION_SZ = int(CNN_FM_SZ / REGION_CNT)


class Image:
    def __init__(self, path):
        img = image.load_img(path)
        self.size = img.size
        self.width = img.width
        self.height = img.height
        self._orig = img
        self.cols = math.ceil(img.width / CNN_INP_SZ)
        self.rows = math.ceil(img.height / CNN_INP_SZ)

        xpad = self.cols * CNN_INP_SZ - img.width
        ypad = self.rows * CNN_INP_SZ - img.height
        padded_img = ImageOps.expand(img, (0,0,xpad,ypad))
        assert padded_img.size == (self.cols * CNN_INP_SZ, self.rows * CNN_INP_SZ), "Padded image size is incorrect"

        self._parts = []
        for r in range(self.rows):
            rows = []
            for c in range(self.cols):
                x = c * CNN_INP_SZ
                y = r * CNN_INP_SZ
                assert x <= padded_img.width - CNN_INP_SZ, "x is out of range"
                assert y <= padded_img.height - CNN_INP_SZ, "y is out of range"
                subimage = padded_img.crop(box=(x,y,x+CNN_INP_SZ,y+CNN_INP_SZ))
                rows.append(subimage)
            assert len(rows) == self.cols, "Number of image cols is incorrect"
            self._parts.append(rows)
        assert len(self._parts) == self.rows, "Number of image rows is incorrect"

    def parts(self, row, col):
        return self._parts[row][col]

    def draw_descriptors(self, descriptors):
        copy = self._orig.copy()

        draw = ImageDraw.Draw(copy)
        for desc in descriptors:
            x,y = desc.kp
            draw.ellipse((x-2,y-2,x+2,y+2), outline=desc.color())

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


def refine_max(x, y, fm):
    assert 0 < x < fm.shape[1]-1, "x is out of range"
    assert 0 < y < fm.shape[0]-1, "y is out of range"

    Dx = 0.5 * (fm[y,x+1] - fm[y,x-1])
    Dy = 0.5 * (fm[y+1,x] - fm[y-1,x])

    Dxx = fm[y,x+1] + fm[y,x-1] - 2*fm[y,x]
    Dyy = fm[y+1,x] + fm[y+1,x] - 2*fm[y,x]
    Dxy = 0.25 * (fm[y+1,x+1] + fm[y-1,x-1] - fm[y+1,x-1] - fm[y-1,x+1])

    A = np.array([[Dxx, Dxy], [Dxy, Dyy]])
    b = np.transpose(np.array([-Dx, -Dy]))
    rmax,_,_,_ = np.linalg.lstsq(A,b,rcond=None)

    rmax = np.clip(rmax, -1, 1)
    return x+rmax[0], y+rmax[1]


class FeatureDetector:
    def __init__(self, threshold):
        self._threshold = threshold
        base = VGG16(weights='imagenet')
        self._model = Model(inputs=base.input, outputs=base.get_layer('block4_conv3').output)

    def produce_featuremaps(self, img):
        assert img.size == (CNN_INP_SZ, CNN_INP_SZ), "Image size does not match CNN"

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        featuremaps = self._model.predict(x)
        return np.squeeze(featuremaps, axis=0)

    def find_keypoints(self, featuremaps):
        width = featuremaps.shape[1]
        height = featuremaps.shape[0]
        keypoints = []
        seen = set()
        for r in range(int(height/REGION_SZ)):
            for c in range(int(width/REGION_SZ)):
                fm_region = featuremaps[r*REGION_SZ:(r+1)*REGION_SZ, c*REGION_SZ:(c+1)*REGION_SZ,:]
                for i,fm in enumerate(np.rollaxis(fm_region,2)):
                    max_idx = np.unravel_index(np.argmax(fm, axis=None), fm.shape)
                    x = max_idx[1] + c*REGION_SZ
                    y = max_idx[0] + r*REGION_SZ
                    if (x,y) not in seen:
                        if fm[max_idx] > self._threshold:
                            seen.add((x,y))
                            if 0 < x < width-1 and 0 < y < height-1:
                                keypoints.append((x,y,i))

        return keypoints

    def create_featuremaps(self, img):
        row_fms = []
        for r in range(img.rows):
            col_fms = []
            for c in range(img.cols):
                fm = self.produce_featuremaps(img.parts(r,c))
                col_fms.append(fm)
            row_fms.append(np.hstack(col_fms))
        full_fm = np.vstack(row_fms)
        assert full_fm.shape == (img.rows*CNN_FM_SZ, img.cols*CNN_FM_SZ, CNN_FM_CNT), "Unexpected feature maps shape"
        return full_fm

    def detect(self, img):
        descriptors = []
        
        featuremaps = self.create_featuremaps(img)
        keypoints = self.find_keypoints(featuremaps)
        for x_kp,y_kp,fm in keypoints:
            x,y = refine_max(x_kp, y_kp, featuremaps[:,:,fm])
            x = int(x*CNN_SCALE + CNN_SCALE/2)
            y = int(y*CNN_SCALE + CNN_SCALE/2)
            if 0 <= x < img.width and 0 <= y < img.height:
                desc = featuremaps[y_kp,x_kp,:]
                descriptors.append(KpDesc((x,y),desc,fm))

        return descriptors


