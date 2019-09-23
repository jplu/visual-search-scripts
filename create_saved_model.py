from classification_models.keras import Classifiers
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda
import tensorflow as tf
from keras import backend as K

ResNet18, _ = Classifiers.get('resnet18')

def preprocess_and_decode(raw_img):
    img = tf.image.decode_jpeg(raw_img, channels=3)
    img = tf.image.resize_images(img, [224, 224], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    
    return img

input_layer = Input(shape=(1,), dtype="string")
output_layer = Lambda(lambda img : tf.map_fn(lambda im : preprocess_and_decode(im[0]), img, dtype="float32"))(input_layer)
raw_model = Model(input_layer, output_layer)

raw_model.summary()

resnet18_model = ResNet18(input_tensor=raw_model.output, weights='imagenet')
avgpooling_layer = resnet18_model.get_layer('pool1').output
final_model = Model(resnet18_model.input, avgpooling_layer)

final_model.summary()

signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={"image": final_model.input}, outputs={"features": final_model.output})
builder = tf.saved_model.builder.SavedModelBuilder('resnet18')

builder.add_meta_graph_and_variables(sess=K.get_session(), tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature})
builder.save()

