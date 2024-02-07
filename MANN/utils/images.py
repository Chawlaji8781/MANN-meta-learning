import numpy as np
from PIL import ImageOps


def get_images_labels(all_images, nb_classes, nb_samples_per_class, image_size, sample_stategy = "uniform"):
    sample_classes = np.random.choice(range(len(all_images)), replace=True, size=nb_classes)
    if sample_stategy == "random":
        labels = np.random.randint(0, nb_classes, nb_classes * nb_samples_per_class)
    elif sample_stategy == "uniform":
        labels = np.concatenate([[i] * nb_samples_per_class for i in range(nb_classes)])
        np.random.shuffle(labels)
    angles = np.random.randint(0, 4, nb_classes) * 90
    images = [image_transform(all_images[sample_classes[i]][np.random.randint(0, len(all_images[sample_classes[i]]))],
                              angle=angles[i]+(np.random.rand()-0.5)*22.5, trans=np.random.randint(-10, 11, size=2).tolist(), size=image_size)
              for i in labels]
    return images, labels

def image_transform(image, angle=0., trans=(0.,0.), size=(20, 20)):
    image = ImageOps.invert(image.convert("L")).rotate(angle, translate=trans).resize(size)
    np_image= np.reshape(np.array(image, dtype=np.float32), newshape=(np.prod(size)))
    max_value = np.max(np_image)
    if max_value > 0.:
        np_image = np_image / max_value
    return np_image
