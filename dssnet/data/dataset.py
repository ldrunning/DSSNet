import os
from glob import glob
import numpy as np
# import openslide
from PIL import Image
from torch.utils.data import dataset
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import cv2


def list_file_tree(path, file_type="tif"):
    image_list = list()
    dir_list = os.listdir(path)
    if os.path.isdir(path):
        image_list += glob(os.path.join(path, "*." + file_type))
    for dir_name in dir_list:
        sub_path = os.path.join(path, dir_name)
        if os.path.isdir(sub_path):
            image_list += list_file_tree(sub_path, file_type)
    return image_list


class CellSeg(dataset.Dataset):
    def __init__(self, data_path, transform=None, augment=None):
        self.data_path = data_path
        self.transform = transform
        self.augment = augment
        self.image_list = list_file_tree(os.path.join(data_path, "img"), "tif")
        self.image_list += list_file_tree(os.path.join(data_path, "img"), "png")
        # self.cyt_list = list_file_tree(os.path.join(data_path, "mask", "cyt"), "tif")
        self.nuc_list = list_file_tree(os.path.join(data_path, "mask"), "tif")
        self.nuc_list += list_file_tree(os.path.join(data_path, "mask"), "png")
        self.label_list = list_file_tree(os.path.join(data_path, "label"), "tif")
        self.label_list += list_file_tree(os.path.join(data_path, "label"), "png")
        # assert len(self.image_list) == len(self.cyt_list)
        self.image_list.sort()
        self.nuc_list.sort()
        self.label_list.sort()
        # assert len(self.image_list) == len(self.nuc_list)

    def __len__(self):
        return np.minimum(len(self.nuc_list), len(self.image_list))

    def __getitem__(self, item):
        filename_img = os.path.split(self.image_list[item])[1]
        filename_nuc = os.path.split(self.nuc_list[item])[1]
        filename_label = os.path.split(self.label_list[item])[1]
        assert filename_nuc == filename_img, filename_img + "!=" + filename_nuc
        # img = Image.open(self.image_list[item]).convert("RGB")  # .resize((2000, 2000))
        # # cyt = Image.open(self.cyt_list[item]).convert("L")
        # nuc = Image.open(self.nuc_list[item]).convert("L")  # .resize((2000, 2000))
        # label = Image.open(self.label_list[item]).convert("L")  # .resize((2000, 2000))
        # cyt = cyt.resize(img.size)

        img = cv2.imread(self.image_list[item], cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        nuc = cv2.imread(self.nuc_list[item], 0)
        label = cv2.imread(self.label_list[item], 0)
        # img = np.array(img)
        # nuc = np.array(nuc)
        # label = np.array(label)
        # nuc = SegmentationMapsOnImage(nuc, shape=nuc.shape)
        # label = SegmentationMapsOnImage(label, shape=nuc.shape)
        if self.transform:
            img, nuc, label = self.transform(img, nuc, label)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.augment:
            img = self.augment(img)
        # nuc = nuc.get_arr()
        # label = label.get_arr()
        # print(nuc.size())
        img = ((np.array(img, dtype=np.float32) - 128) / 128.0).transpose((2, 0, 1))
        # img = img[:, 104:616, 104:616]
        # cyt = (np.array(cyt) > 128).astype(np.float32)
        nuc = np.array(nuc).astype(np.int64)[np.newaxis, :, :]
        label = np.array(label).astype(np.int64)[np.newaxis, :, :]
        # nuc = np.minimum(nuc, 1)

        return img, nuc, label


class SingleImage(dataset.Dataset):
    def __init__(self, data_path, transform=None, augment=None):
        self.data_path = data_path
        self.transform = transform
        self.augment = augment
        self.image_list = list_file_tree(os.path.join(data_path), "png")
        # assert len(self.image_list) == len(self.cyt_list)
        self.image_list.sort()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img = Image.open(self.image_list[item])  # .resize((2000, 2000))
        img = ((np.array(img, dtype=np.float32) - 128) / 128.0).transpose((2, 0, 1))
        return img


class SlideDataSet(dataset.Dataset):
    def __init__(self, filename, level, step, read_size, out_size):
        """

        :param filename: openslide文件名
        :param level: 缩放等级
        :param step: 在缩放后图像上的步长，即在原图上的步长为step*2^level
        :param size: 输出图像的大小
        """
        self.root_dir = filename
        self.read_size = read_size
        self.out_size = out_size
        self.slide_image = openslide.OpenSlide(filename)
        self.level = level
        self.step = (step[0] * (2 ** level), step[1] * (2 ** level))
        self.dimensions = self.slide_image.dimensions
        self.start_piont, self.end_piont = self.get_image_region()
        self.len_x = int((self.end_piont[0] - self.start_piont[0]) / self.step[0])
        self.len_y = int((self.end_piont[1] - self.start_piont[1]) / self.step[1])

    def get_image_region(self):
        step_x = 100
        step_y = 1000
        precision = 10
        start_piont_x = step_x
        start_piont_y = step_y
        [end_piont_x, end_piont_y] = self.dimensions
        end_piont_x -= step_x
        end_piont_y -= step_y
        while True:
            piont = self.slide_image.read_region((end_piont_x, end_piont_y), self.level, (1, 1)).convert("RGB")
            piont = np.array(piont)
            if np.max(piont) > 0:
                while True:
                    piont2 = self.slide_image.read_region((int(self.dimensions[0] / 2), end_piont_y), self.level,
                                                          (1, 1)).convert("RGB")
                    piont2 = np.array(piont2)
                    if np.max(piont2) == 0:
                        break
                    else:
                        end_piont_y += precision
                end_piont_y -= precision
                while True:
                    piont2 = self.slide_image.read_region((end_piont_x, int(self.dimensions[1] / 2)), self.level,
                                                          (1, 1)).convert("RGB")
                    piont2 = np.array(piont2)
                    if np.max(piont2) == 0:
                        break
                    else:
                        end_piont_x += precision
                end_piont_y -= precision
                break
            else:
                end_piont_x -= step_x
                end_piont_y -= step_y
                end_piont_x = np.maximum(end_piont_x, int(self.dimensions[0] / 2))
                end_piont_y = np.maximum(end_piont_y, int(self.dimensions[1] / 2))

        while True:
            piont = self.slide_image.read_region((start_piont_x, start_piont_y), self.level, (1, 1)).convert("RGB")
            piont = np.array(piont)
            if np.max(piont) > 0:
                while True:
                    piont2 = self.slide_image.read_region((int(self.dimensions[0] / 2), start_piont_y), self.level,
                                                          (1, 1)).convert(
                        "RGB")
                    piont2 = np.array(piont2)
                    if np.max(piont2) == 0:
                        break
                    else:
                        start_piont_y -= precision
                start_piont_y += precision
                while True:
                    piont2 = self.slide_image.read_region((start_piont_x, int(self.dimensions[1] / 2)), self.level,
                                                          (1, 1)).convert("RGB")
                    piont2 = np.array(piont2)
                    if np.max(piont2) == 0:
                        break
                    else:
                        start_piont_x -= precision
                start_piont_x += precision
                break
            else:
                start_piont_x += step_x
                start_piont_y += step_y
                start_piont_x = np.minimum(start_piont_x, int(self.dimensions[0] / 2))
                start_piont_y = np.minimum(start_piont_y, int(self.dimensions[1] / 2))

        return (start_piont_x, start_piont_y), (end_piont_x, end_piont_y)

    def __len__(self):
        return self.len_x * self.len_y

    def read_region(self, position, level, read_size):
        image = self.slide_image.read_region(position, level, read_size).convert("RGB")
        return np.array(image)

    def __getitem__(self, idx):
        """

        :param idx:
        :return: 返回相应的level的图像，以及该图像在原始分辨率上的坐标
        """

        position_x = int(idx % self.len_x) * self.step[0] + self.start_piont[0]
        position_y = int(idx / self.len_x) * self.step[1] + self.start_piont[1]
        image = self.slide_image.read_region((position_x, position_y),
                                             self.level, self.read_size).convert("RGB").resize(self.out_size)
        position = np.array([position_x, position_y]).reshape([1, 2])
        image = np.array(image, dtype=np.float32)
        image = ((np.array(image, dtype=np.float32) - 128) / 128.0).transpose((2, 0, 1))
        return image.astype(np.float32), position


if __name__ == '__main__':
    data = CellSeg("/home/cell_segmention")
    one, two = data[0]
    print(len(data))
