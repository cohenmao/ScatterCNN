import hickle
import numpy
import matplotlib.pyplot as plt

with open('/home/ubuntu/DL/Images/ImageNet/Output30/tar_root_dir/val_hkl_b64_b_64/0000.hkl', 'r') as f:
    imagenet_train_batch = hickle.load(f)

with open('/home/ubuntu/DL/Images/ImageNet/Output30/tar_root_dir/labels/val_labels.npy', 'r') as f:
    imagenet_train_labels = numpy.load(f)



with open('/home/ubuntu/DL/Images/ImageNet/OutputKylberg/tar_root_dir/val_hkl_b32_b_32/0000.hkl', 'r') as f:
    kylberg_train_batch = hickle.load(f)

with open('/home/ubuntu/DL/Images/ImageNet/OutputKylberg/tar_root_dir/labels/val_labels.npy', 'r') as f:
    kylberg_train_labels = numpy.load(f)

imagenet_idx = [ii for ii in range(len(imagenet_train_labels)) if imagenet_train_labels[ii] == 0]
imagenet_image = imagenet_train_batch[:, :, :, imagenet_idx[0]]
imagenet_image = numpy.transpose(imagenet_image, (1, 2, 0))
plt.imshow(imagenet_image)

kylberg_idx = [ii for ii in range(len(kylberg_train_labels)) if kylberg_train_labels[ii] == 0]
kylberg_image = kylberg_train_batch[0, :, :, kylberg_idx[0]]
plt.imshow(kylberg_image, cmap='gray')

1