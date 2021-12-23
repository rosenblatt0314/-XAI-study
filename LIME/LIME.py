from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray
from skimage.util import montage

from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()

# 이미지를 흑백으로 만들고 LIME이 처리할 수 있는 형태로 변환하기
X_vec = np.stack([gray2rgb(iimg) for iimg in faces.data.reshape((-1, 64, 64))], 0)
y_vec = faces.target.astype(np.uint8)

fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
ax1.imshow(montage(X_vec[:, :, :, 0]), 
           cmap='gray', 
           interpolation='none')
ax1.set_title('All Faces')
ax1.axis('off')


X_train, X_test, y_train, y_test = train_test_split(X_vec, 
                                                    y_vec, 
                                                    train_size=0.70)

class PipeStep(object):
    """
        Wrapper for tuning functions into pipeline transforms (no-fitting)
    """
    def __init__(self, step_func):
        self._step_func = step_func
        
    def fit(self, *args):
        return self
    
    def transform(self, X):
        return self._step_func(X)
    
makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

simple_pipeline = Pipeline([
    ('Make Gray', makegray_step),
    ('Flatten Image', flatten_step),
    ('Normalize', Normalizer()), # 전처리 과정에 노멀라이저 추가
    ('MLP', MLPClassifier(activation='relu', 
                          hidden_layer_sizes=(400, 40), 
                          random_state=1))
    ])

simple_pipeline.fit(X_train, y_train)


pipe_pred_test = simple_pipeline.predict(X_test)
pipe_pred_prop = simple_pipeline.predict_proba(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, 
                            y_pred=pipe_pred_test))

simple_pipeline = Pipeline([
    ('Make Gray', makegray_step),
    ('Flatten Image', flatten_step),
    ('Normalize', Normalizer()),
    ('MLP', MLPClassifier(
        activation='relu',
        alpha=1e-7,
        epsilon=1e-6,
        hidden_layer_sizes=(800, 120),
        random_state=1))    
])

simple_pipeline.fit(X_train, y_train)



explainer = lime_image.LimeImageExplainer()

# 이미지 분할 알고리즘: quickshift(기본), slic, selzenszwalb
segmenter = SegmentationAlgorithm('slic',
                                  n_segments=100,
                                  compactness=1,
                                  sigma=1)

olivetti_test_index = 0
exp = explainer.explain_instance(np.float64(X_test[olivetti_test_index]), 
                                 top_labels=6, 
                                 num_samples=10000, 
                                 segmentation_fn=segmenter,
                                 classifier_fn=simple_pipeline.predict_proba,
                                 ) 