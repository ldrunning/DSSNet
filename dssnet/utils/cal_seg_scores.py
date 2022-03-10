import os
import shutil
from glob import glob

import cv2
import numpy as np
from skimage import measure, color


def list_file_tree(path, file_title="", file_type="tif"):
    image_list = list()
    dir_list = os.listdir(path)
    if os.path.isdir(path):
        image_list += glob(os.path.join(path, file_title + "*." + file_type))
    for dir_name in dir_list:
        sub_path = os.path.join(path, dir_name)
        if os.path.isdir(sub_path):
            image_list += list_file_tree(sub_path, file_title, file_type)
    return image_list


def do_somethings(paths):
    img_path, gt_path = paths
    # cv2.imshow("img_ori", img)
    mask_gt = cv2.imread(gt_path)
    # img_lab = color.rgb2lab(cv2.imread(img_path)[:, :, ::-1])
    labels_pos = measure.label(mask_gt[:, :, 2] > 128, connectivity=2)
    # labels_all = measure.label(mask_gt[:, :, 0] < 128, connectivity=2)
    # dilate_kernel = np.ones((7, 7), np.uint8)
    total_scores = 0
    for i in range(labels_pos.max()):
        if np.sum(labels_pos == (i + 1)) > 50:
            # num = labels_all[labels_pos == (i + 1)][100]
            # temp_label = (labels_all == num).astype(np.uint8)
            temp_label = (labels_pos == (i + 1)).astype(np.uint8)
            edges = cv2.Canny(temp_label * 255, threshold1=20, threshold2=50)
            y, x = np.nonzero(edges)  # 注：矩阵的坐标系表示和绘图用的坐标系不同，故需要做坐标转换
            edge_list = np.array([[_x, _y] for _x, _y in zip(x, y)])  # 边界点坐标
            best_ellipse = cv2.fitEllipse(edge_list)  # 椭圆拟合
            size_ellipse = np.pi * best_ellipse[1][0] * best_ellipse[1][1] / 4
            size_label = temp_label.sum()
            scores = (mask_gt[temp_label > 0, 2]).mean() / 255
            # dilation_label = cv2.dilate(temp_label, dilate_kernel, iterations=2) - temp_label
            # if 15 < img_lab[dilation_label > 0, 1].mean() < 60 and size_label < 1000:
            #     scores *= 2
            similarity_ellipse = size_label / size_ellipse
            if similarity_ellipse > 0.98:
                # if scores < 0.62:
                #     scores = 0
                if size_label / 1000 > total_scores:
                    total_scores = size_label / 1000
                # cv2.imshow("img", img)
                # cv2.imshow("temp_label", mask_gt[:, :, 2] * temp_label)
                # print(size_label, similarity_ellipse, scores, total_scores)
                # cv2.imshow("mask_gt", mask_gt)
                # cv2.waitKey(0)

    return total_scores


def move_top_regions(dir_root):
    save_dir = os.path.join(dir_root, "top_regions")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    dirlist = os.listdir(dir_root)
    for dirname in dirlist:
        path = os.path.join(dir_root, dirname, "top22")
        save_path = os.path.join(save_dir, dirname)
        os.makedirs(save_path, exist_ok=True)
        if os.path.exists(path):
            shutil.move(path, save_path)


def top_regions(dataset, dir_root, top_n=22):
    save_path = os.path.join(dir_root, "top_regions")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    files_img = list_file_tree(os.path.join(dir_root, "images"), "", "png")
    files_gt = list_file_tree(os.path.join(dir_root, "masks"), "", "png")
    files_img.sort()
    files_gt.sort()
    print(len(files_img), files_img[0])
    scores_list = []
    for paths in zip(files_img, files_gt):
        scores_list.append([do_somethings(paths), paths])
    scores_list = sorted(scores_list, key=lambda p: p[0], reverse=True)
    top_list = []
    for scores, paths in scores_list:
        if scores == 0:
            continue
        img_name = os.path.split(paths[0])[1]
        content = img_name[:-4].split("_")
        x, y = int(content[0]), int(content[1])
        bbox = np.array([[x, y, x + 700, y + 700]])
        overlapping = False
        for scores_, paths_ in top_list:
            img_name_ = os.path.split(paths_[0])[1]
            content_ = img_name_[:-4].split("_")
            x_, y_ = int(content_[0]), int(content_[1])
            bbox_ = np.array([[x_, y_, x_ + 700, y_ + 700]])
            iou = bbox_cover_iou(bbox, bbox_) > 0.2
            if iou.sum() > 0:
                overlapping = True
        if not overlapping:
            top_list.append([scores, paths])
        if len(top_list) == top_n:
            break

    for i, (scores, paths) in enumerate(top_list):
        img_path, gt_path = paths
        img_name = os.path.split(img_path)[1]
        content = img_name[:-4].split("_")
        x, y = int(content[0]), int(content[1])
        img = np.array(dataset.slide_image.read_region((x - 350, y - 350), 0, (1400, 1400)).convert("RGB"))
        cv2.rectangle(img, (350, 350), (1050, 1050), (255, 0, 0), thickness=3)
        cv2.imwrite(os.path.join(save_path, img_name), img[:, :, ::-1])
        # gt_name = os.path.split(gt_path)[1]
        # shutil.copy(img_path, os.path.join(save_path, str(i) + "_" + str(scores) + "_" + img_name))
        # shutil.copy(gt_path, os.path.join(save_path, str(i) + "_" + str(scores) + "_" + gt_name))


def bbox_cover_iou(bbox_x, bbox_gt):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_x (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_gt (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_x.shape[1] != 4 or bbox_gt.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_x[:, None, :2], bbox_gt[:, :2])
    # bottom right
    br = np.minimum(bbox_x[:, None, 2:], bbox_gt[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_x = np.prod(bbox_x[:, 2:] - bbox_x[:, :2], axis=1)
    # area_b = xp.prod(bbox_gt[:, 2:] - bbox_gt[:, :2], axis=1)
    return area_i / (area_x[:, None])


def read_gt(path):
    label_list = open(path, "r").read().splitlines()
    class_name = ("ascus", "lsil", "hsil", "lisl", "hisl")
    bbox_list = list()
    for labe_line in label_list:
        content = labe_line.rstrip('\n').split(", ")
        label_name = content[0].lower()
        if label_name in class_name:
            if label_name == class_name[0]:
                label_num = 1
            elif label_name == class_name[1] or label_name == class_name[3]:
                label_num = 2
            else:
                label_num = 3
            x_min = float(content[1])
            y_min = float(content[2])
            x_max = float(content[3]) + x_min
            y_max = float(content[4]) + y_min

            bbox_list.append(np.array([label_num, x_min, y_min, x_max, y_max], dtype=np.int))
    return bbox_list


def read_predict(path):
    files_img = list_file_tree(path, "*img", "png")
    files_img.sort()
    pred_bbox = []
    for file in files_img:
        img_name = os.path.split(file)[1]
        content = img_name[:-4].split("_")
        score, x, y = float(content[1]), int(content[3]), int(content[4])
        pred_bbox.append([score, x, y, x + 427, y + 427])
    return np.array(pred_bbox)


def top_k_recall(abnormal_path, gt_path):
    gt_list = list_file_tree(gt_path, file_type="txt")
    gt_list.sort()
    all_recall = []
    all_gt = []
    recall_sum = 0
    for gt_file in gt_list:
        file_name = os.path.split(gt_file)[1]
        predict_file_name = file_name[:-8]
        predict_file = os.path.join(abnormal_path, predict_file_name, "top22")

        if os.path.exists(predict_file):
            top_bboxs = read_predict(predict_file)
            gt_bbox = np.array(read_gt(gt_file))
            iou = bbox_cover_iou(top_bboxs[:, 1:], gt_bbox[:, 1:])
            recall_bbox_index = np.max(iou, axis=1) > 0
            gt_bbox_index = np.max(iou, axis=0) > 0
            # gt_bbox_recall = gt_bbox[gt_bbox_index]
            # top_bboxs[recall_bbox_index, 0] = 100
            files_img = list_file_tree(predict_file, "*img", "png")
            files_mask = list_file_tree(predict_file, "*mask", "png")
            files_mask.sort()
            files_img.sort()
            for img_path, mask_path, index in zip(files_img, files_mask, recall_bbox_index):
                if index:
                    os.rename(img_path, img_path[:-4] + "_good" + img_path[-4:])
                    os.rename(mask_path, mask_path[:-4] + "_good" + mask_path[-4:])

            recall_num = np.sum(gt_bbox_index)
            # recall_num = len(abnormal_bbox)
            recall_sum += recall_num
            all_recall.append({file_name[:-8]: recall_num})
            all_gt.append({file_name[:-8]: len(gt_bbox)})
            print(recall_num)
    print(recall_sum)
    return all_recall, all_gt


if __name__ == '__main__':
    slide_root = "/media/khtao/workplace/WorkCenter/2019-10/resnet50/result/tongji5th_less400/分类模型"
    dir_list = os.listdir(slide_root)
    for dir in dir_list:
        dir_root = os.path.join(slide_root, dir)
        save_path = os.path.join(dir_root, "top_regions")
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
        files_img = list_file_tree(os.path.join(dir_root), "", "png")
        files_gt = list_file_tree(os.path.join(dir_root), "", "jpg")
        files_img.sort()
        files_gt.sort()
        print(len(files_img), files_img[0])
        scores_list = []
        for paths in zip(files_img, files_gt):
            scores_list.append([do_somethings(paths), paths])
        scores_list = sorted(scores_list, key=lambda p: p[0], reverse=True)
        top_list = scores_list[:10]
        for i, (scores, paths) in enumerate(top_list):
            file_name = os.path.split(paths[0])[1]
            shutil.copy(paths[0], os.path.join(save_path, file_name))

# dir_root = "result/07301929_0.8513"
# top_regions(dir_root, 10)
# top_k_recall(dir_root,
#              "/home/khtao/MD3400-2/Public_Data/original_data/our-system/sfy5_svs_hologic/label_txt")
