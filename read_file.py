import struct

def return_image (dir_img, dir_label, index):
    #accepts image idx and returns [number, [list of pixels]
    with open(dir_img, mode="rb") as file_img:
        images = file_img.read()
    with open(dir_label, mode="rb") as file_img2:
        labels = file_img2.read()
    label_idx = index + 8
    image_idx = index * (28 * 28) + 16
    label = struct.unpack(">B", labels[label_idx:label_idx+1])
    image = struct.unpack(">" + 28 * 28 * "B", images[image_idx:image_idx + (28 * 28)])
    return [label, image]


def return_images (dir_img, dir_label):
    #   Converts into [[(image by pixel), number], [], ...]
    with open(dir_img, mode="rb") as file_img:
        images = file_img.read()
    images_header = struct.unpack(">4I", images[:16])
    print(images_header)

    with open(dir_label, mode="rb") as file_img2:
        labels = file_img2.read()
    train_labels_header = struct.unpack(">2I", labels[:8])

    labels_all = struct.unpack(">" + "B" * (len(labels) - 8), labels[8:])
    images_all = struct.unpack(">" + "B" * (len(images) - 16), images[16:])

    result = []
    for idx1 in range(images_header[1]):
        num_and_img = []
        image = []
        for idx2 in range(28*28):
            image.append(images_all[idx1+idx2])
        img_tuple = tuple(image)
        num_and_img.append(img_tuple)
        num_and_img.append(labels_all[idx1])
        result.append(num_and_img)

    return result

# print(return_image('./mnist/train-images.idx3-ubyte', './mnist/train-labels.idx1-ubyte', 0))
# print(return_image('./mnist/t10k-images.idx3-ubyte', './mnist/t10k-labels.idx1-ubyte', 0))