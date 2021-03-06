import numpy as np
import tensorflow as tf
import yolo.config as cfg
import cv2
slim = tf.contrib.slim


class YOLONet(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.output_size = (self.cell_size * self.cell_size) *\
            (self.num_class + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA
        #offset.shape=[7,7,2],7个7行2列的矩阵
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
        #输入图像为[None,height,width,3]
        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')
        #获取预测结果logitsshape=[None,7*7*(20+2*5)=1470]
        self.logits = self.build_network(
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training)

        if is_training:
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class])
            #计算loss
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn=leaky_relu(alpha),
                weights_regularizer=slim.l2_regularizer(0.0005),
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            ):
                #images:[None,448,448,3],net:[None,453,453,3]
                net = tf.pad(
                    images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                    name='pad_1')
                net = slim.conv2d(
                    net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                #上一步的net结果为[None,14,14,1024],pad后为[None,16,16,1024]
                net = tf.pad(
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')
                net = slim.conv2d(
                    net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                #转置后的shape为[None,1024,7,7]
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                #flatten后shape=[none,50176]，flatten将输入扁平化但保留batch_size,假设第一维是batch。
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
                net = slim.fully_connected(
                    net, num_outputs, activation_fn=None, scope='fc_36')
        return net

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
        #截取V使之在min和max之间
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)
    #predict=[None,1470],label=[None,7,7,25]
    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            #[None,7,7,20]
            predict_classes = tf.reshape(
                predicts[:, :self.boundary1],#predict前面7*7*20=980个元素为预测的类别概率
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            #[None,7,7,2]
            predict_scales = tf.reshape(#中间的7*7*2=98个元素为每个栅格对应的两个iou
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            #predict_boxes.shape=[None,7,7,2,4]
            predict_boxes = tf.reshape(#最后7*7*8个数作为两个box预测值
                predicts[:, self.boundary2:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            # label.shape=[None, self.cell_size, self.cell_size, 5 + self.num_class])
            #response:置信度[None,7,7,1]
            response = tf.reshape(#label中的iou
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])
            #[None,7,7,1,4]
            boxes = tf.reshape(#label中的每个栅格对应的rect
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            #boxes中心坐标和长宽归一化到[0,1]，shape=[None,7,7,2,4]
            boxes = tf.tile(
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            #[None,7,7,2,20]
            classes = labels[..., 5:]

            #self.offset=[7,7,2]转为[1,7,7,2]
            offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            #下面offset=[None,7,7,2]
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            #offset_tran.shape=[None,7,7,2]
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))
            #predict_boxes.shape=[None,7,7,2,4]
            with tf.Session() as sess:
                temp = sess.run(offset)
                temp1 = sess.run(offset_tran)
                print("ok")
            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + offset) / self.cell_size,
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size,
                 tf.square(predict_boxes[..., 2]),
                 tf.square(predict_boxes[..., 3])], axis=-1)
            #注意：label中box为（真实x、真实y,真实width,height),这些数组在上面除image_size归一化为[0.1]
            #iou_predict_truth：[0,1]
            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # iou_predict_truth.shape=[BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # tf.reduce_max找出iou最大的那个box，shape=[None,7,7,1],tf.cast返回[None,7,7,2]
            #response中全是0或1，表示是否有目标
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask
            #box为title后的shape=[None,7,7,2,4]
            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss,计算类别损失
            class_delta = response * (predict_classes - classes)#[None,7,7,1]*[None,7,7,20]
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale

            # object_loss，iou的损失函数
            #predict_scales.shape=[None,7,7,2],
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss，1-iou
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss,坐标位置损失函数
            # predict_boxes.shape=[None,7,7,2,4],boxes_tran.shape=[None,7,7,2,4],
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)


def leaky_relu(alpha):
    def op(inputs):
        #return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
        return tf.nn.relu(inputs, name='relu')
    return op
