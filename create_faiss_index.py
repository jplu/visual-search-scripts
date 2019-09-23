import numpy as np
import faiss
import multiprocessing
import glob
import os
import sys
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
NUM_CPU = multiprocessing.cpu_count() // 4
manager = multiprocessing.Manager()
features = manager.list()

class Network():
    def __init__(self):
        from classification_models.keras import Classifiers
        from keras.models import Model
        from keras.layers import Input, Lambda
        import tensorflow as tf

        def preprocess_and_decode(raw_img):
            img = tf.image.decode_jpeg(raw_img, channels=3)
            img = tf.image.resize_images(img, [224, 224], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
            img = tf.image.central_crop(img, central_fraction=0.7)

            return img

        ResNet18, _ = Classifiers.get('resnet18')
        input_layer = Input(shape=(1,), dtype="string")
        output_layer = Lambda(lambda img : tf.map_fn(lambda im : preprocess_and_decode(im[0]), img, dtype="float32"))(input_layer)
        raw_model = Model(input_layer, output_layer)
        resnet18_model = ResNet18(input_tensor=raw_model.output, weights='imagenet')
        avgpooling_layer = resnet18_model.get_layer('pool1').output
        self.model = Model(resnet18_model.input, avgpooling_layer)
    
    def predict(self, img):
        return self.model.predict(img)


def featurize(images):
    model = Network()
    count = 0

    for image in images:
        idx = np.asarray([int(image.split('/')[1].split('.')[0])])
 
        try:
            with open(image, "rb") as img_file:
                img_array = np.asarray([img_file.read()])
                ebd = model.predict(img_array)
                features.append((ebd[0], idx[0]))
        except tf.errors.InvalidArgumentError as e:
            print("Error with", image)
            sys.stdout.flush()
            os.remove(image)
        
        count += 1
    assert count == len(images)

index = faiss.index_factory(512, "IDMap,Flat")
list_files = glob.glob(os.path.join('images', '**.jpg'))
slices = [list_files[i::NUM_CPU] for i in range(NUM_CPU)]
jobs = []

for i, s in enumerate(slices):
    j = multiprocessing.Process(target=featurize, args=(s,))
    jobs.append(j)
    j.start()

for j in jobs:
    j.join()

features = zip(*features)
vectors, ids = list(features)

del features

vectors = np.asarray(list(vectors))
ids = np.asarray(list(ids))

index.add_with_ids(vectors, ids)
faiss.write_index(index, "index_amazon.faiss")

