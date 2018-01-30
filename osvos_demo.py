import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import osvos
from dataset import Dataset


class OSVOSDemo(object):

    def __init__(self, is_train, train_iters, seq_name,
                 result_path="results/Segmentations/480p/OSVOS", davis_root="/home/z840/ALISURE/Data/DAVIS",
                 model_result_path="models/OSVOS_demo", parent_model="models/OSVOS_parent/OSVOS_parent.ckpt-50000"):
        self.seq_name = seq_name

        # 参数
        self.is_train = is_train
        self.train_iters = train_iters

        # 模型
        self.model_result_path = os.path.join(model_result_path, self.seq_name)
        self.parent_model = parent_model

        # 结果
        self.result_path = os.path.join(result_path, self.seq_name)

        # 数据
        self.images_path = os.path.join(davis_root, 'DAVIS/JPEGImages/480p', self.seq_name)
        self.annotations_path = os.path.join(davis_root, 'DAVIS/Annotations/480p', self.seq_name)
        self.dataset, self.test_images = self._get_data()

        pass

    def _get_data(self):
        test_frames = sorted(os.listdir(self.images_path))
        test_images = [os.path.join(self.images_path, frame) for frame in test_frames]
        if self.is_train:
            train_images = [os.path.join(self.images_path, '00000.jpg') + ' ' + os.path.join(self.annotations_path, '00000.png'),
                            os.path.join(self.images_path, '00020.jpg') + ' ' + os.path.join(self.annotations_path, '00020.png'),
                            os.path.join(self.images_path, '00040.jpg') + ' ' + os.path.join(self.annotations_path, '00040.png'),
                            os.path.join(self.images_path, '00060.jpg') + ' ' + os.path.join(self.annotations_path, '00060.png'),
                            os.path.join(self.images_path, '00080.jpg') + ' ' + os.path.join(self.annotations_path, '00080.png')]
            dataset = Dataset(train_images, test_images, './', data_aug=True)
        else:
            dataset = Dataset(None, test_images, './')
        return dataset, test_images

    def train_model(self, learning_rate=1e-8, side_supervision=3, display_step=10):
        if self.is_train:
            with tf.Graph().as_default():
                global_step = tf.Variable(0, name='global_step', trainable=False)
                osvos.train_finetune(self.dataset, self.parent_model, side_supervision, learning_rate,
                                     self.model_result_path, self.train_iters, self.train_iters, display_step,
                                     global_step, iter_mean_grad=1, ckpt_name=self.seq_name)
            pass
        pass

    def test_model(self):
        with tf.Graph().as_default():
            checkpoint_path = "{}/{}.ckpt-{}".format(self.model_result_path, self.seq_name, self.train_iters)
            osvos.test(self.dataset, checkpoint_path, self.result_path)
        pass

    def show_result(self):
        overlay_color = [255, 0, 0]
        transparency = 0.6
        plt.ion()
        for img_p in self.test_images:
            frame_num = os.path.basename(img_p).split('.')[0]
            img = np.array(Image.open(os.path.join(self.images_path, img_p)))
            mask = np.array(Image.open(os.path.join(self.result_path, "{}.png".format(frame_num))))
            mask = mask / np.max(mask)
            im_over = np.ndarray(img.shape)
            im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0] * transparency + (1 - transparency) * img[:, :, 0])
            im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (overlay_color[1] * transparency + (1 - transparency) * img[:, :, 1])
            im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2] * transparency + (1 - transparency) * img[:, :, 2])
            plt.imshow(im_over.astype(np.uint8))
            plt.axis('off')
            plt.show()
            plt.pause(0.01)
            plt.clf()
        pass

    pass


if __name__ == '__main__':

    osvos_demo = OSVOSDemo(is_train=True, train_iters=500, seq_name="bear")

    osvos_demo.train_model()
    osvos_demo.test_model()

    osvos_demo.show_result()

    pass
