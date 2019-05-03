import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='js_visualize_dataset.py')
    parser.add_argument('--csv', type=str, required=True)
    opt = parser.parse_args()

    assert os.path.isfile(opt.csv)
    dir_name = os.path.dirname(opt.csv)

    labels = {}
    with open(opt.csv, 'r') as f:
        for i, csv_line in enumerate(f.readlines()):
            if i == 0:
                continue
            filename, width, height, classs, xmin, ymin, xmax, ymax = csv_line.strip().split(',')
            xmin,ymin,xmax,ymax, width, height = int(xmin), int(ymin), int(xmax), int(ymax), int(width), int(height)
            #print(filename,width,height,classs,xmin,ymin,xmax,ymax)
            #
            x = ((xmin + xmax) / 2.0) / width
            y = ((ymin + ymax) / 2.0) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            #print(x, y, w, h)
            #
            if filename not in labels:
                labels[filename] = []
            labels[filename].append("0 %f %f %f %f\n" % (x, y, w, h))
    #
    for fname, lbls in labels.items():
        pre, ext = os.path.splitext(fname)
        #print(os.path.join(dir_name, pre + ".txt"))
        with open(os.path.join(dir_name, pre + ".txt"), 'w') as f:
            for lbl in lbls:
                f.write(lbl)
    #
    print('%d files label created' % len(labels))