from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from model import Model
import numpy as np
import math
from PIL import ImageDraw, ImageOps


class CNN:
    LAYER = 13
    FM_SZ = 28
    FM_CNT = 512

    INP_SZ = 224
    SCALE = INP_SZ / FM_SZ
    FM_BORDER = math.ceil(LAYER/SCALE)
    BORDER = FM_BORDER * SCALE
    FM_EFF_SZ = FM_SZ - 2*FM_BORDER
    EFF_SZ = INP_SZ - 2*BORDER

    @classmethod
    def calc_parts(cls, size):
        parts = math.ceil((size - 2*cls.BORDER) / cls.EFF_SZ)
        padding = parts*cls.EFF_SZ + 2*cls.BORDER - size
        return (parts, int(padding))


REGION_CNT = 6
REGION_SZ = int(CNN.FM_SZ / REGION_CNT)


class Image:
    def __init__(self, path):
        img = image.load_img(path)
        self.size = img.size
        self.width = img.width
        self.height = img.height
        self._orig = img

        self.cols,xpad = CNN.calc_parts(self.width)
        self.rows,ypad = CNN.calc_parts(self.height)
        self._padded_img = ImageOps.expand(img, (0,0,xpad,ypad))
        assert self._padded_img.width == (self.cols*CNN.EFF_SZ + 2*CNN.BORDER), "Padded image width is incorrect"
        assert self._padded_img.height == (self.rows*CNN.EFF_SZ + 2*CNN.BORDER), "Padded image height is incorrect"

    def part(self, row, col):
        assert 0 <= row < self.rows, "Incorrect row"
        assert 0 <= col < self.cols, "Incorrect col"
        x = col * CNN.EFF_SZ
        y = row * CNN.EFF_SZ
        assert x <= self._padded_img.width - CNN.INP_SZ, "x is out of range"
        assert y <= self._padded_img.height - CNN.INP_SZ, "y is out of range"
        return self._padded_img.crop(box=(x,y,x+CNN.INP_SZ,y+CNN.INP_SZ))

    def inside(self, x, y):
        return CNN.BORDER <= x < self.width-CNN.BORDER and CNN.BORDER <= y < self.height-CNN.BORDER

    def draw_descriptors(self, descriptors):
        copy = self._orig.copy()

        draw = ImageDraw.Draw(copy)
        for desc in descriptors:
            x,y = desc.kp()
            draw.ellipse((x-2,y-2,x+2,y+2), outline=desc.color())

        return copy


class Keypoint:
    def __init__(self, x, y, fm, val):
        self.x = x
        self.y = y
        self.fm = fm
        self.val = val

    def __repr__(self):
        return f'({self.x},{self.y},{self.fm},{self.val})'


class KpDesc:
    def __init__(self, kp, desc, fm):
        self._kp = kp
        self.desc = desc
        self._fm = fm

    def kp(self):
        return self._kp

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


class Region:
    def __init__(self):
        self._keypoints = []

    def add_keypoint(self, new_kp):
        if self._keypoints == []:
            self._keypoints.append(new_kp)
        for i,cur_kp in enumerate(self._keypoints):
            if (new_kp.x, new_kp.y) == (cur_kp.x, cur_kp.y):
                if new_kp.val > cur_kp.val:
                    cur_kp = new_kp
                break
            if (new_kp.x, new_kp.y) < (cur_kp.x, cur_kp.y):
                self._keypoints.insert(i, new_kp)
                break
        
    def get_top_keypoints(self, nr):
        kps = sorted(self._keypoints, key=lambda kp: kp.val, reverse=True)
        return kps[0:nr]


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
    def __init__(self, threshold, kp_number):
        self._threshold = threshold
        self._kp_number = kp_number
        base = VGG16(weights='imagenet')
        self._model = Model(inputs=base.input, outputs=base.get_layer('block4_conv3').output)

    def produce_featuremaps(self, img):
        assert img.size == (CNN.INP_SZ, CNN.INP_SZ), "Image size does not match CNN"

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        featuremaps = self._model.predict(x)
        return np.squeeze(featuremaps, axis=0)

    def create_regions(self, featuremaps, img_sz):
        width = featuremaps.shape[1]
        height = featuremaps.shape[0]
        regions = []
        for r in range(math.floor((height-2*CNN.FM_BORDER)/REGION_SZ)):
            for c in range(math.floor((width-2*CNN.FM_BORDER)/REGION_SZ)):
                rx = c*REGION_SZ + CNN.FM_BORDER
                ry = r*REGION_SZ + CNN.FM_BORDER
                if rx > img_sz[0] or ry > img_sz[1]:
                    continue

                fm_region = featuremaps[ry:ry+REGION_SZ, rx:rx+REGION_SZ,:]
                region = Region()
                for i,fm in enumerate(np.rollaxis(fm_region,2)):
                    max_idx = np.unravel_index(np.argmax(fm, axis=None), fm.shape)
                    max_val = fm[max_idx]
                    x = max_idx[1] + rx
                    y = max_idx[0] + ry
                    if max_val > self._threshold:
                        if 0 < x < width-1 and 0 < y < height-1:
                            region.add_keypoint(Keypoint(x,y,i,max_val))
                regions.append(region)

        return regions

    def create_featuremaps(self, img):
        row_fms = []
        for r in range(img.rows):
            col_fms = []
            for c in range(img.cols):
                fm = self.produce_featuremaps(img.part(r,c))
                if r == 0:
                    if c == 0:
                        col_fms.append(fm[:-CNN.FM_BORDER, :-CNN.FM_BORDER])
                    elif c == img.cols-1:
                        col_fms.append(fm[:-CNN.FM_BORDER, CNN.FM_BORDER:])
                    else:
                        col_fms.append(fm[:-CNN.FM_BORDER, CNN.FM_BORDER:-CNN.FM_BORDER])
                elif r == img.rows-1:
                    if c == 0:
                        col_fms.append(fm[CNN.FM_BORDER:, :-CNN.FM_BORDER])
                    elif c == img.cols-1:
                        col_fms.append(fm[CNN.FM_BORDER:, CNN.FM_BORDER:])
                    else:
                        col_fms.append(fm[CNN.FM_BORDER:, CNN.FM_BORDER:-CNN.FM_BORDER])
                else:
                    if c == 0:
                        col_fms.append(fm[CNN.FM_BORDER:-CNN.FM_BORDER, :-CNN.FM_BORDER])
                    elif c == img.cols-1:
                        col_fms.append(fm[CNN.FM_BORDER:-CNN.FM_BORDER, CNN.FM_BORDER:])
                    else:
                        col_fms.append(fm[CNN.FM_BORDER:-CNN.FM_BORDER, CNN.FM_BORDER:-CNN.FM_BORDER])
                # col_fms.append(fm[CNN.FM_BORDER:-CNN.FM_BORDER, CNN.FM_BORDER:-CNN.FM_BORDER])
            row_fms.append(np.hstack(col_fms))
        full_fm = np.vstack(row_fms)
        # assert full_fm.shape == (img.rows*CNN.FM_EFF_SZ, img.cols*CNN.FM_EFF_SZ, CNN.FM_CNT), "Unexpected feature maps shape"
        return full_fm

    def detect(self, img):
        descriptors = []
        
        featuremaps = self.create_featuremaps(img)
        regions = self.create_regions(featuremaps, img.size)

        for region in regions:
            for kp in region.get_top_keypoints(self._kp_number):
                x,y = refine_max(kp.x, kp.y, featuremaps[:,:,kp.fm])
                x = int(x*CNN.SCALE + CNN.SCALE/2)
                y = int(y*CNN.SCALE + CNN.SCALE/2)
                if img.inside(x, y):
                    desc = featuremaps[kp.y,kp.x,:]
                    descriptors.append(KpDesc((x,y),desc,kp.fm))

        return descriptors


