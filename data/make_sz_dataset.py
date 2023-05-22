import os.path
import shutil
import cv2
import numpy as np


# sky RGB is [0,0,0], and ground RGB is [153, 108, 6]
Ground_RGB = np.array([153, 108, 6], dtype=np.uint8)
Sky_RGB = np.array([0, 0, 0], dtype=np.uint8)


def make_dataset_from_copying(src_dir: str, dst_dir: str, mode='train',
                              vis_contour=True, min_box_len=0.02):
    prefix_dir = f'{src_dir}/{mode}'
    output_prefix_dir = f'{dst_dir}/{mode}'
    print(f'source directory is {prefix_dir}\n'
          f'target directory is {output_prefix_dir}.')
    if not os.path.exists(output_prefix_dir):
        os.makedirs(output_prefix_dir)
    img_ind = 0
    while True:
        # 复制图片
        img_name = f'{prefix_dir}/{img_ind}_scene.png'
        if not os.path.exists(img_name):
            break
        output_img_name = f'{output_prefix_dir}/{img_ind}_scene.png'
        shutil.copyfile(img_name, output_img_name)
        # 创建物体label文件
        label_content = []
        seg_name = f'{prefix_dir}/{img_ind}_seg.png'
        color_img: np.ndarray = cv2.imread(seg_name)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        H, W, _ = color_img.shape
        img_pixels = color_img.reshape(H * W, 3)
        unique_colors, inv_indices = np.unique(img_pixels, return_inverse=True, axis=0)
        # 求得每个像素的ID
        inv_indices = inv_indices.reshape(H, W)
        for px_id in range(unique_colors.shape[0]):
            bi_mat = np.zeros((H, W), np.uint8)
            bi_mat[inv_indices == px_id] = 255
            # 求得每个不相邻的轮廓
            contours, _ = cv2.findContours(bi_mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 检查
            if vis_contour:
                vis_img = color_img.copy()
                cv2.drawContours(vis_img, contours, -1, (255,255,255))
                cv2.imshow('contours', vis_img)
                cv2.waitKey()
                cv2.destroyAllWindows()
            rgb = unique_colors[px_id, :]
            for contour in contours:
                length = cv2.arcLength(contour, closed=True)
                if length < 2 * min_box_len * max(H, W):
                    # 过滤掉太小的东西
                    continue
                if np.all(rgb == Sky_RGB):
                    label_line = f'0 '
                elif np.all(rgb == Ground_RGB):
                    label_line = f'1 '
                else:
                    label_line = f'2 '
                n, _, _ = contour.shape
                for i in range(n):
                    x, y = contour[i, 0, 0], contour[i, 0, 1]
                    x /= W
                    y /= H
                    label_line += f'{x:.8f} {y:.8f}'
                label_content.append(label_line)
        # 创建标签文件
        output_label_name = f'{output_prefix_dir}/{img_ind}_scene.txt'
        with open(output_label_name, 'w') as f:
            for line in label_content:
                f.write(f'{line}\n')
        img_ind += 1
    print(f'Handle {img_ind} pictures.')


if __name__ == '__main__':
    src_dir = r'E:\Py_Projects\estimate-surface-normal-uncertainty\data_split'
    make_dataset_from_copying(src_dir, '.', 'train', False)
    make_dataset_from_copying(src_dir, '.', 'test', False)
    pass