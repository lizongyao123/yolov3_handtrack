import argparse
import glob
import os

import cv2
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='js_visualize_dataset.py')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--same_folder', action='store_true')
    opt = parser.parse_args()

    #
    if os.path.isdir(opt.data):
        img_files = []
        for e in ['*.bmp', '*.jpg', '*.png']:
            img_files += glob.glob(os.path.join(opt.data, e))
        if opt.same_folder:
            label_files = []
            for x in img_files:
                pre, ext = os.path.splitext(x)
                label_files.append(os.path.join(pre + ".txt"))
        else:
            label_files = [
                x.replace('images', 'labels').replace('.bmp', '.txt').replace('.jpg', '.txt').replace('.png', '.txt')
                    for x in img_files]
        pause_time = 10000
    elif os.path.isfile(opt.data):
        img_files = [opt.data]
        if opt.same_folder:
            pre, ext = os.path.splitext(opt.data)
            label_files = [os.path.join(pre + ".txt")]
        else:
            label_files = [
                opt.data.replace('images', 'labels').replace('.bmp', '.txt').replace('.jpg', '.txt').replace('.png', '.txt')]
        pause_time = 0
    else:
        raise FileNotFoundError(opt.data)
    
    #
    for img_file, label_file in zip(img_files, label_files):
        assert os.path.isfile(img_file), img_file
        #assert os.path.isfile(label_file), label_file
        #print(img_file, label_file)
        
        # read image
        img = cv2.imread(img_file) # BGR
        h, w = img.shape[0], img.shape[1]

        # read labels
        if os.path.isfile(label_file):
            with open(label_file, 'r') as f:
                x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                print(x)
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2)
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2)
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2)
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2)
                for label in labels:
                    c1, c2 = (int(label[1]), int(label[2])), (int(label[3]), int(label[4]))
                    cv2.rectangle(img, c1, c2, [225, 255, 255], thickness=1)

        #
        cv2.imshow('dataset visualize test', img)
        if cv2.waitKey(pause_time) == 27: # ESC
            break

    #
    cv2.destroyAllWindows()