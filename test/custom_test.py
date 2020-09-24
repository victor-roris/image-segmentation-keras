import numpy as np
import tempfile
import time
from keras_segmentation.models import all_models
from keras_segmentation.data_utils.data_loader import \
    verify_segmentation_dataset, image_segmentation_generator
from keras_segmentation.predict import batch_prediction, predict, evaluate, predict_multiple

tr_im = "/home/victor/SWDevelopment/projects/DATASPARTAN/ml_models/image-segmentation-keras/test/example_dataset/images_prepped_train"
tr_an = "/home/victor/SWDevelopment/projects/DATASPARTAN/ml_models/image-segmentation-keras/test/example_dataset/annotations_prepped_train"
te_im = "/home/victor/SWDevelopment/projects/DATASPARTAN/ml_models/image-segmentation-keras/test/example_dataset/images_prepped_test"
te_im = "/home/victor/SWDevelopment/projects/DATASPARTAN/ai-documents/data/export/image"
te_an = "/home/victor/SWDevelopment/projects/DATASPARTAN/ml_models/image-segmentation-keras/test/example_dataset/annotations_prepped_test"

import glob
import os
import tensorflow as tf


def test_gpu():
    sess = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
    print("GPU available? ", sess)
    built = tf.test.is_built_with_cuda()
    print("tf is built with CUDA? ", built)
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    print("Num GPUs used: ", len(gpus))
    print("Num CPUs used: ", len(cpus))
    print(tf.sysconfig.get_build_info())


def test_batch_prediction():
    model_name = "fcn_8"
    h = 224
    w = 256
    n_c = 100
    check_path = tempfile.mktemp()
    print(f"Trained model stored in the tmp folder : {check_path}")


    m = all_models.model_from_name[model_name](n_c, input_height=h, input_width=w)

    m.train(train_images=tr_im,
            train_annotations=tr_an,
            steps_per_epoch=2,
            epochs=2,
            checkpoints_path=check_path
            )

    m.train(train_images=tr_im,
            train_annotations=tr_an,
            steps_per_epoch=2,
            epochs=2,
            checkpoints_path=check_path,
            augmentation_name='aug_geometric', do_augment=True
            )

    m.predict_segmentation(np.zeros((h, w, 3))).shape

    # check_path = "/tmp/tmpufnaqffb"

    inps = glob.glob(os.path.join(te_im, "*.*"))

    # Batch prediction
    start_time = time.time()
    bpredictions = batch_prediction(inp_dir=te_im, checkpoints_path=check_path, out_dir="/tmp")
    print(f"--- Batch predictions of {len(inps)} images in %s seconds ---" % (time.time() - start_time))

    # Instance by instance prediction
    start_time = time.time()
    mpredictions = predict_multiple(inp_dir=te_im, checkpoints_path=check_path, out_dir="/tmp")
    print(f"--- Instance by instance predictions of {len(inps)} images in  %s seconds ---" % (time.time() - start_time))

    for idx in range(0, len(bpredictions)):
        bpred = bpredictions[idx]
        mpred = mpredictions[idx]

        comparison = bpred == mpred
        equal_arrays = comparison.all()
        assert equal_arrays is True, "Different predictions by batch or by for instance"

    # --- Batch predictions of 3221 images in 421.86106848716736 seconds ---
    # --- Instance by instance predictions of 3221 images in  650.1733341217041 seconds ---


if __name__ == "__main__":
    test_gpu()
    test_batch_prediction()
