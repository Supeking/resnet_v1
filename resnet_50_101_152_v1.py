import tensorflow as tf
import utils
import numpy as np
import scipy.io as sio
import cv2


class Resnet:
    def __init__(self, gpu=0, cls=2, res_name='resnet101', checkpoint_dir='./model', model_name='test',
                 lr=1e-5):
        self.cls = cls
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.res_name = res_name
        self.lr = lr
        CFG = {'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3], 'resnet152': [3, 8, 36, 3]}
        self.resnet50_path = 'weights\\imgnet_resnet101.npz'
        if gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()
        self.build(CFG[res_name])

    def branch1(self, x, numOut, s):
        with tf.variable_scope("conv1"):
            conv1 = utils.relu(utils.Bn(utils.conv2d(x, numOut / 4, d_h=s, d_w=s), training=self.is_training))
        with tf.variable_scope("conv2"):
            conv2 = utils.relu(utils.Bn(utils.conv2d(conv1, numOut / 4, 3, 3), training=self.is_training))
        with tf.variable_scope("conv3"):
            conv3 = utils.Bn(utils.conv2d(conv2, numOut), training=self.is_training)
        return conv3

    def branch2(self, x, numOut, s):
        with tf.variable_scope("convshortcut"):
            return utils.Bn(utils.conv2d(x, numOut, d_h=s, d_w=s), training=self.is_training)

    def residual(self, x, numOut, stride=1, name='res'):
        with tf.variable_scope(name):
            block = self.branch1(x, numOut, stride)
            if x.get_shape().as_list()[3] == numOut:
                return utils.relu(tf.add_n([x, block]))
            else:
                skip = self.branch2(x, numOut, stride)
                return utils.relu(tf.add_n([block, skip]))

    def inference(self, x, grah):
        with tf.variable_scope("conv0"):
            if self.res_name == "resnet50":
                net = utils.relu(utils.Bn(utils.conv2d(x, 64, 7, 7, 2, 2, bias=True), training=self.is_training))
            else:
                net = utils.relu(utils.Bn(utils.conv2d(x, 64, 7, 7, 2, 2, bias=False), training=self.is_training))
        with tf.name_scope("pool1"):
            net = utils.max_pool(net, 3, 3, 2, 2)
        with tf.variable_scope("group0"):
            for i in range(grah[0]):
                net = self.residual(net, 256, name='block' + str(i))
        with tf.variable_scope("group1"):
            for i in range(grah[1]):
                if i == 0:
                    net = self.residual(net, 512, 2, name='block' + str(i))
                else:
                    net = self.residual(net, 512, name='block' + str(i))
        with tf.variable_scope("group2"):
            for i in range(grah[2]):
                if i == 0:
                    net = self.residual(net, 1024, 2, name='block' + str(i))
                else:
                    net = self.residual(net, 1024, name='block' + str(i))
        with tf.variable_scope("group3"):
            for i in range(grah[3]):
                if i == 0:
                    net = self.residual(net, 2048, 2, name='block' + str(i))
                else:
                    net = self.residual(net, 2048, name='block' + str(i))
        with tf.name_scope("pool5"):
            net = utils.global_pool(net)
        with tf.variable_scope("linear"):
            net = tf.nn.dropout(net, keep_prob=self.keep_prob)
            net = utils.linear(net, 1000)
        return net

    def load_weight(self):
        param = dict(np.load(self.resnet50_path))
        vars = tf.global_variables(scope="inference")
        for v in vars:
            nameEnd = v.name.split('/')[-1]
            if nameEnd == "moving_mean:0":
                name = v.name[10:-13] + "mean/EMA"
            elif nameEnd == "moving_variance:0":
                name = v.name[10:-17] + "variance/EMA"
            else:
                name = v.name[10:-2]
            if name == 'linear/W':
                param[name] = param['linear/W'].reshape(2048, 1000)
            self.sess.run(v.assign(param[name]))
            print("Copy weights: " + name + "---->" + v.name)

    def build(self, CFG):
        self.inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], "inputs")
        self.labels = tf.placeholder(tf.int32, [None], "labels")
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        with tf.variable_scope("inference"):
            self.outs = self.inference(self.inputs, CFG)
        # 定义损失函数
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.outs, labels=self.labels))
        with tf.variable_scope('minimizer'):
            adam = tf.train.AdamOptimizer(self.lr)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optm = adam.minimize(self.loss)
        # 模型保存
        self.saver = tf.train.Saver(max_to_keep=2)
        # 初始化
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # 加载预训练权重
        self.load_weight()

    def train(self):
        # 检测是否有保存的模型
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        # 数据读取
        
        # 开始训练

    def test(self):
        pp_mean = sio.loadmat('pp_mean.mat')['normalization']
        pp_mean_224 = pp_mean[16:-16, 16:-16, :]
        img = cv2.imread('C:\\Users\\sk\\Desktop\\test\\ILSVRC2012_val_00000001.JPEG').astype('float32')
        im = img[75:-76, 138:-138, :]
        im = im - pp_mean_224
        im = np.reshape(im, (1, 224, 224, 3))
        A = self.sess.run(self.outs, feed_dict={self.inputs: im})
        print(A[0].argsort()[-10:][::-1])
        print('-------')
        
    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(
            checkpoint_dir, self.model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            return True
        else:
            return False


def main():
    model = Resnet()
    model.test()


if __name__ == "__main__":
    main()
