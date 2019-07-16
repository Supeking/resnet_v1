import tensorflow as tf
import utils
import numpy as np
import scipy.io as sio
import cv2

class Resnet50:
    def __init__(self, gpu=0, cls=2, checkpoint_dir='./model', model_name='test', lr=1e-5):
        self.cls = cls
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.lr = lr
        self.resnet50_path = 'weights\\imgnet_resnet50.npz'
        if gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()
        self.build()

    def branch1(self, x, numOut, s):
        with tf.variable_scope("conv1"):
            conv1 = utils.relu(utils.Bn(utils.conv2d(x, numOut/4, d_h=s, d_w=s), training=self.is_training))
        with tf.variable_scope("conv2"):
            conv2 = utils.relu(utils.Bn(utils.conv2d(conv1, numOut/4, 3, 3), training=self.is_training))
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

    def inference(self, x):
        with tf.variable_scope("conv0"):
            conv1 = utils.relu(utils.Bn(utils.conv2d(x, 64, 7, 7, 2, 2, bias=True), training=self.is_training))
        with tf.name_scope("pool1"):
            pool1 = utils.max_pool(conv1, 3, 3, 2, 2)
        with tf.variable_scope("group0"):
            res2a = self.residual(pool1, 256, name='block0')
            res2b = self.residual(res2a, 256, name='block1')
            res2c = self.residual(res2b, 256, name='block2')
        with tf.variable_scope("group1"):
            res3a = self.residual(res2c, 512, 2, name='block0')
            res3b = self.residual(res3a, 512, name='block1')
            res3c = self.residual(res3b, 512, name='block2')
            res3d = self.residual(res3c, 512, name='block3')
        with tf.variable_scope("group2"):
            res4a = self.residual(res3d, 1024, 2, name='block0')
            res4b = self.residual(res4a, 1024, name='block1')
            res4c = self.residual(res4b, 1024, name='block2')
            res4d = self.residual(res4c, 1024, name='block3')
            res4e = self.residual(res4d, 1024, name='block4')
            res4f = self.residual(res4e, 1024, name='block5')
        with tf.variable_scope("group3"):
            res5a = self.residual(res4f, 2048, 2, name='block0')
            res5b = self.residual(res5a, 2048, name='block1')
            res5c = self.residual(res5b, 2048, name='block2')
        with tf.name_scope("pool5"):
            pool5 = utils.global_pool(res5c)
        with tf.variable_scope("linear"):
            dropout = tf.nn.dropout(pool5, keep_prob=self.keep_prob)
            out = utils.linear(dropout, 1000)
        return out

    def load_weight(self):
        param = dict(np.load(self.resnet50_path))
        vars = tf.global_variables(scope= "inference")
        for v in vars:
            nameEnd = v.name.split('/')[-1]
            if nameEnd == "moving_mean:0":
                name =  v.name[10:-13]+"mean/EMA"
            elif nameEnd == "moving_variance:0":
                name = v.name[10:-17]+"variance/EMA"
            else:
                name = v.name[10:-2]
            if name == 'linear/W':
                param[name] = param['linear/W'].reshape(2048, 1000)
            self.sess.run(v.assign(param[name]))
            print("Copy weights: " + name + "---->"+ v.name)

    def build(self):
        self.inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], "inputs")
        self.labels = tf.placeholder(tf.int32, [None], "labels")
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        with tf.variable_scope("inference"):
            self.outs = self.inference(self.inputs)
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
        print('---;;;;;;;----')
       
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
    model = Resnet50()
    model.test()

if __name__ == "__main__":
    main()
