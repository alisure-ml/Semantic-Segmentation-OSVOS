import os
import osvos
import tensorflow as tf
from dataset import Dataset


class OSVOSParentDemo(object):

    def __init__(self, train_parent_list="DAVIS/train_parent.txt",
                 train_data_root="/home/z840/ALISURE/Data/DAVIS/DAVIS-2017-trainval-480p/DAVIS",
                 imagenet_ckpt="models/vgg_16.ckpt", model_result_path="models/OSVOS_parent",
                 # train_iters=list([15, 30, 50]), boundaries=list([10, 15, 25, 30, 40]),
                 # store_memory=True, data_aug=True, display_step=1, save_step=100):
                 train_iters=list([15000, 30000, 50000]), boundaries=list([10000, 15000, 25000, 30000, 40000]),
                 store_memory=True, data_aug=True, display_step=100, save_step=5000):

        # 模型
        self.imagenet_ckpt = imagenet_ckpt
        self.model_result_path = model_result_path
        self.ckpt_name = os.path.basename(self.model_result_path)

        # 参数
        self.store_memory = store_memory
        self.data_aug = data_aug
        self.display_step = display_step
        self.save_step = save_step
        self.train_iters = train_iters

        # 学习率相关
        self.iter_mean_grad = 10
        self.init_learning_rate = 1e-8
        self.boundaries = boundaries
        self.values = [self.init_learning_rate, self.init_learning_rate * 0.1,
                       self.init_learning_rate, self.init_learning_rate * 0.1,
                       self.init_learning_rate, self.init_learning_rate * 0.1]

        # 数据
        self.dataset = Dataset(train_parent_list, None, train_data_root, self.store_memory, self.data_aug)

        pass

    def train(self):

        # dsn_2_loss + dsn_3_loss + dsn_4_loss + dsn_5_loss + main_loss
        with tf.Graph().as_default():
            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = tf.train.piecewise_constant(global_step, self.boundaries, self.values)
            osvos.train_parent(self.dataset, self.imagenet_ckpt, 1, learning_rate, self.model_result_path,
                               self.train_iters[0], self.save_step, self.display_step, global_step,
                               iter_mean_grad=self.iter_mean_grad, test_image_path=None, ckpt_name=self.ckpt_name)

        # 0.5 * (dsn_2_loss + dsn_3_loss + dsn_4_loss + dsn_5_loss) + main_loss
        with tf.Graph().as_default():
            global_step = tf.Variable(self.train_iters[0], name='global_step', trainable=False)
            learning_rate = tf.train.piecewise_constant(global_step, self.boundaries, self.values)
            osvos.train_parent(self.dataset, self.imagenet_ckpt, 2, learning_rate, self.model_result_path,
                               self.train_iters[1], self.save_step, self.display_step, global_step,
                               iter_mean_grad=self.iter_mean_grad, resume_training=True,
                               test_image_path=None, ckpt_name=self.ckpt_name)

        # main_loss
        with tf.Graph().as_default():
            global_step = tf.Variable(self.train_iters[1], name='global_step', trainable=False)
            learning_rate = tf.train.piecewise_constant(global_step, self.boundaries, self.values)
            osvos.train_parent(self.dataset, self.imagenet_ckpt, 3, learning_rate, self.model_result_path,
                               self.train_iters[2], self.save_step, self.display_step, global_step,
                               iter_mean_grad=self.iter_mean_grad, resume_training=True,
                               test_image_path=None, ckpt_name=self.ckpt_name)
        pass

    pass


if __name__ == '__main__':

    osvos_parent_demo = OSVOSParentDemo()
    osvos_parent_demo.train()
