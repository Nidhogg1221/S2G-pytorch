"""
pytorch implementation of the model .py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

MAX_DEGREE = 6

"""
diff between pytorch and tensorflow 
    
    1. order of tensor 
    tensor in pytorch: (batch_size, channels, height, width)， 
    tensor in TensorFlow: (batch_size, height, width, channels)。
    axis 3 in tensorflow = dim 1 in pytorch.
    
    2. different usage of forward method in pytorch and tensorflow

"""


class Sat2GraphModel(nn.Module):
    def __init__(self, image_size=352, image_ch=3, downsample_level=1, batch_size=8, resnet_step=8, channel=12,
                 mode="train", joint_with_seg=True, lr=0.001):
        super(Sat2GraphModel, self).__init__()

        self.image_size = image_size
        self.image_ch = image_ch  # lgb, 3
        self.channel = channel  # used in the cnn
        self.joint_with_seg = joint_with_seg
        self.mode = mode
        self.batch_size = batch_size
        self.resnet_step = resnet_step  # 8 depth of resnet
        self.lr = lr

        self.train_seg = False
        self.is_training = True

        """layer definition"""
        # conv1: 3 -> 12, input_shape: [2, 3, 352, 352] → [2, 12, 352, 352]
        self.conv1 = ConvLayer(image_ch, channel, self.is_training, 5, 5, 1, 1, padding="2", batchnorm=False)
        # conv2: 12 -> 24,  input_shape: [2, 12, 352, 352] → [2, 24, 176, 176]
        self.conv2 = ConvLayer(channel, channel * 2, self.is_training, 5, 5, 2, 2, padding="2", batchnorm=True)

        # sequential block 1, x_4s: 24 -> 48, input_shape: [2, 24, 176, 176] → [2, 48, 88, 88]
        self.x_4s = nn.Sequential(
            self.class_reduce_block(channel * 2, channel * 4),
            self.class_resnet_blocks(channel * 4, int(self.resnet_step / 8))
        )
        # sequential block 2, x_8s: 48 -> 96,  input_shape: [2, 48, 88, 88] → [2, 96, 44, 44]
        self.x_8s = nn.Sequential(
            self.class_reduce_block(channel * 4, channel * 8),
            self.class_resnet_blocks(channel * 8, int(self.resnet_step / 4))
        )

        # sequential block 3, x_16s: 96 -> 192, input_shape: [2, 96, 44, 44] → [2, 192, 22, 22]
        self.x_16s = nn.Sequential(
            self.class_reduce_block(channel * 8, channel * 16),
            self.class_resnet_blocks(channel * 16, int(self.resnet_step / 2))
        )

        # sequential block 4, x_32s: 192 -> 384, input_shape: [2, 192, 22, 22] → [2, 384, 11, 11]
        self.x_32s = nn.Sequential(
            self.class_reduce_block(channel * 16, channel * 32),
            self.class_resnet_blocks(channel * 32, self.resnet_step)
        )

        # [2, 48, 176, 176] → [2, 96, 88, 88] → [2, 192, 44, 44] → [2, 384, 22, 22]
        self.a1_2s = ClassAggregateBlock(channel * 2, channel * 4, channel * 4, self.is_training)
        # conv2 + x_4s, [2, 24, 176, 176] + [2, 48, 88, 88] → [2, 48, 88, 88]
        self.a1_4s = ClassAggregateBlock(channel * 4, channel * 8, channel * 8, self.is_training)
        self.a1_8s = ClassAggregateBlock(channel * 8, channel * 16, channel * 16, self.is_training)

        # [2, 48, 176, 176]
        self.a1_16s_agg = ClassAggregateBlock(channel * 16, channel * 32, channel * 32, self.is_training)
        self.a1_16s = self.class_resnet_blocks(channel * 32, int(self.resnet_step / 2))

        # [2, 24, 176, 176] → [2, 48, 88, 88] → [2, 96, 44, 44]
        self.a2_2s = ClassAggregateBlock(channel * 4, channel * 8, channel * 4, self.is_training)
        self.a2_4s = ClassAggregateBlock(channel * 8, channel * 16, channel * 8, self.is_training)

        # [2, 96, 44, 44]
        self.a2_8s_agg = ClassAggregateBlock(channel * 16, channel * 32, channel * 16)
        self.a2_8s = self.class_resnet_blocks(channel * 16, int(self.resnet_step / 4))

        # [2, 48, 176, 176]
        self.a3_2s = ClassAggregateBlock(channel * 4, channel * 8, channel * 4, self.is_training)

        # [2, 96, 88, 88]
        self.a3_4s_agg = ClassAggregateBlock(channel * 8, channel * 16, channel * 8, self.is_training)
        self.a3_4s = self.class_resnet_blocks(channel * 8, int(resnet_step / 8))

        # [2, 96, 176, 176]
        self.a4_2s_agg = ClassAggregateBlock(channel * 4, channel * 8, channel * 8, self.is_training)

        # Final Convolution Layer
        # [2, 48, 176, 176]
        self.a5_2s = ConvLayer(channel * 8, channel * 4, self.is_training, 3, 3, 1, 1, batchnorm=True)

        # [2, 48, 352, 352]
        self.a_out_agg = ClassAggregateBlock(channel, channel * 4, channel * 4, self.is_training)

        self.a_out = ConvLayer(channel * 4, 2 + MAX_DEGREE * 4 + (2 if self.joint_with_seg else 0),
                               self.is_training, 3,
                               3, 1, 1, batchnorm=False,
                               activation='linear')

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.writer = SummaryWriter()

    def class_reduce_block(self, in_ch, out_ch, resnet_step=0, k=3):
        layers = nn.Sequential(
            ConvLayer(in_ch, out_ch, self.is_training, 3, 3, 1, 1, batchnorm=True, padding='1'),
            ConvLayer(out_ch, out_ch, self.is_training, 3, 3, 2, 2, batchnorm=True, padding='1')
        )
        return layers

    def class_resnet_blocks(self, channel, resnet_step=0):
        layers = []
        if resnet_step > 0:
            for i in range(resnet_step):
                layers.append(ResBlock(in_channels=channel, out_channels=channel, downsample=False))
            layers.append(nn.BatchNorm2d(channel))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    #########################################################################################
    def forward(self, inputdata):
        conv1 = self.conv1(inputdata)
        conv2 = self.conv2(conv1)
        # conv1.shape:  torch.Size([2, 12, 352, 352])
        # conv2.shape:  torch.Size([2, 24, 176, 176])

        x_4s = self.x_4s(conv2)
        x_8s = self.x_8s(x_4s)
        x_16s = self.x_16s(x_8s)
        x_32s = self.x_32s(x_16s)
        # x_4s.shape:  torch.Size([2, 48, 88, 88])
        # x_8s.shape:  torch.Size([2, 96, 44, 44])
        # x_16s.shape:  torch.Size([2, 192, 22, 22])
        # x_32s.shape:  torch.Size([2, 384, 11, 11])

        a1_2s = self.a1_2s(conv2, x_4s)
        a1_4s = self.a1_4s(x_4s, x_8s)
        a1_8s = self.a1_8s(x_8s, x_16s)
        a1_16s_agg = self.a1_16s_agg(x_16s, x_32s)
        a1_16s = self.a1_16s(a1_16s_agg)

        # a1_2s.shape: torch.Size([2, 48, 176, 176])
        # conv2.shape: torch.Size([2, 24, 176, 176]) + x_4s.shape: torch.Size([2, 48, 88, 88])

        # a1_4s.shape: torch.Size([2, 96, 88, 88])
        # a1_8s.shape: torch.Size([2, 192, 44, 44])
        # a1_16s_agg.shape: torch.Size([2, 384, 22, 22])
        # a1_16s.shape: torch.Size([2, 384, 22, 22])

        a2_2s = self.a2_2s(a1_2s, a1_4s)
        a2_4s = self.a2_4s(a1_4s, a1_8s)
        a2_8s_agg = self.a2_8s_agg(a1_8s, a1_16s)
        a2_8s = self.a2_8s(a2_8s_agg)

        # a2_2s.shape:  torch.Size([2, 48, 176, 176])
        # a2_4s.shape:  torch.Size([2, 96, 88, 88])
        # a2_8s_agg.shape:  torch.Size([2, 192, 44, 44])
        # a2_8s.shape:  torch.Size([2, 192, 44, 44])

        a3_2s = self.a3_2s(a2_2s, a2_4s)
        a3_4s_agg = self.a3_4s_agg(a2_4s, a2_8s)
        a3_4s = self.a3_4s(a3_4s_agg)

        # a3_2s.shape:  torch.Size([2, 48, 176, 176])
        # a3_4s_agg.shape:  torch.Size([2, 96, 88, 88])
        # a3_4s.shape:  torch.Size([2, 96, 88, 88])

        a4_2s_agg = self.a4_2s_agg(a3_2s, a3_4s)

        a5_2s = ConvLayer(in_channels=self.channel * 8, out_channels=self.channel * 4,
                          is_training=self.is_training, kx=3, ky=3,
                          stride_x=1, stride_y=1, batchnorm=True, padding='1')(a4_2s_agg).to(a4_2s_agg.device)

        a_out_agg = self.a_out_agg(conv1, a5_2s)

        a_out = ConvLayer(in_channels=self.channel * 4,
                          out_channels=2 + MAX_DEGREE * 4 + (2 if self.joint_with_seg else 0),
                          is_training=self.is_training, kx=3, ky=3,
                          stride_x=1, stride_y=1, batchnorm=False,
                          padding='1', activation='linear')(a_out_agg).to(a_out_agg.device)

        # a4_2s.shape: torch.Size([2, 96, 176, 176])
        # a4_2s_agg.shape: torch.Size([2, 96, 176, 176])
        # a5_2s.shape: torch.Size([2, 48, 176, 176])
        # a_out_agg.shape: torch.Size([2, 48, 352, 352])

        # a_out.shape: torch.Size([2, 28, 352, 352])  2 + 24 + 2

        return a_out

    #########################################################################################

    """modified"""

    """
    tensor shape in pytorch: (batch_size, channels, height, width)，
    tensor shape in tensorflow: (batch_size, height, width, channels)。
    we need to unstack/concat with dim=1
    """

    def unstack(self, tensor, dim=1, size=None):
        ts = torch.unbind(tensor, dim=dim)

        new_ts = []

        for t in ts:
            new_shape = [-1, 1, self.image_size, self.image_size] if size is None else [-1, 1, size, size]
            new_ts.append(t.view(*new_shape))

        return new_ts

    """modified, down sample part omitted"""
    """
    clip_by_value → torch.clamp
    
    tf.losses.softmax_cross_entropy computes the cross-entropy between logits (unprocessed by softmax) and one-hot 
    encoded labels. 
    
    nn.CrossEntropyLoss computes the cross-entropy between logits (unprocessed by softmax) and class 
    index labels. PyTorch's nn.CrossEntropyLoss expects target labels in class index form, not in one-hot encoded 
    form. one-hot encoding
    """

    def SupervisedLoss(self, imagegraph_output, imagegraph_target_prob, imagegraph_target_vector, batch_size=2):
        imagegraph_outputs = self.unstack(imagegraph_output, dim=1)  # length of imagegraph_outputs: 28
        imagegraph_target_probs = self.unstack(imagegraph_target_prob, dim=1)  # length of imagegraph_target_probs: 14
        imagegraph_target_vectors = self.unstack(imagegraph_target_vector,
                                                 dim=1)  # length of imagegraph_target_vectors: 12

        # create soft mask with shape [2, 1, 352, 352]
        soft_mask = torch.clamp(imagegraph_target_probs[0] - 0.01, 0.0, 0.99) + 0.01
        soft_mask2 = soft_mask.view(batch_size, self.image_size, self.image_size)

        keypoint_prob_loss = 0
        #  the softmax cross-entropy loss between the target keypoint_prob map and the model predictions
        keypoint_prob_output = torch.cat(imagegraph_outputs[0:2], dim=1)
        keypoint_prob_target = torch.cat(imagegraph_target_probs[0:2], dim=1)
        keypoint_prob_target_indices = torch.argmax(keypoint_prob_target, dim=1)
        keypoint_prob_loss = nn.CrossEntropyLoss()(keypoint_prob_output, keypoint_prob_target_indices)

        direction_prob_loss = 0
        # the loss between multiple direction probabilities, where a softmax cross-entropy loss is calculated for
        # each direction and averaged across all directions according to MAX_DEGREE(6).

        for i in range(MAX_DEGREE):
            prob_output = torch.cat(imagegraph_outputs[2 + i * 4: 2 + i * 4 + 2], dim=1)
            prob_target = torch.cat(imagegraph_target_probs[2 + i * 2: 2 + i * 2 + 2], dim=1)
            prob_target_indices = torch.argmax(prob_target, dim=1)

            loss = nn.CrossEntropyLoss(reduction='none')(prob_output, prob_target_indices)
            direction_prob_loss += (soft_mask2 * loss).mean()

        direction_prob_loss /= MAX_DEGREE

        direction_vector_loss = 0
        # 计算每个方向向量的均方误差 (MSE) 损失，并按 MAX_DEGREE 平均

        for i in range(MAX_DEGREE):
            vector_output = torch.cat(imagegraph_outputs[2 + i * 4 + 2: 2 + i * 4 + 4], dim=1)
            vector_target = torch.cat(imagegraph_target_vectors[i * 2:i * 2 + 2], dim=1)
            direction_vector_loss += (soft_mask * (vector_output - vector_target) ** 2).mean()
        direction_vector_loss /= MAX_DEGREE

        if self.joint_with_seg:
            # Convert one-hot encoded targets to class indices for PyTorch

            seg_target_indices = torch.argmax(self.input_seg_gt_target, dim=1)

            seg_loss = nn.CrossEntropyLoss()(
                torch.cat([imagegraph_outputs[2 + MAX_DEGREE * 4], imagegraph_outputs[2 + MAX_DEGREE * 4 + 1]], dim=1),
                seg_target_indices)
            return keypoint_prob_loss, direction_prob_loss * 10.0, direction_vector_loss * 1000.0, seg_loss * 0.1
        else:
            return keypoint_prob_loss, direction_prob_loss * 10.0, direction_vector_loss * 1000.0, keypoint_prob_loss - keypoint_prob_loss

    """
    The shape of the tensor after processing with the Merge method is: [batch_size, 2 + 4 × MAX_DEGREE, image_size, image_size].
    batch_size: The batch size, which remains unchanged.
    image_size: The width and height of the image, which remain unchanged.
    2 + 4 * 6: (MAX_DEGREE): The number of channels after merging.
    """

    def Merge(self, imagegraph_target_prob, imagegraph_target_vector):
        imagegraph_target_probs = self.unstack(imagegraph_target_prob, dim=1)
        imagegraph_target_vectors = self.unstack(imagegraph_target_vector, dim=1)

        new_list = []
        new_list += imagegraph_target_probs[0:2]

        for i in range(MAX_DEGREE):
            new_list += imagegraph_target_probs[2 + i * 2:2 + i * 2 + 2]
            new_list += imagegraph_target_vectors[i * 2:i * 2 + 2]
        return torch.cat(new_list, dim=1)

    def SoftmaxOutput(self, imagegraph_output):
        # length of imagegraph_outputs: 26, shape: [2, 352, 352]
        imagegraph_outputs = self.unstack(imagegraph_output, dim=1)

        new_outputs = [torch.sigmoid(imagegraph_outputs[0] - imagegraph_outputs[1])]
        new_outputs.append(1.0 - new_outputs[-1])

        for i in range(MAX_DEGREE):
            new_outputs.append(torch.sigmoid(imagegraph_outputs[2 + i * 4] - imagegraph_outputs[2 + i * 4 + 1]))
            new_outputs.append(1.0 - new_outputs[-1])

            # new_outputs[-1]: [2, 2, 352, 352]？
            new_outputs.append(torch.cat(imagegraph_outputs[2 + i * 4 + 2:2 + i * 4 + 4], dim=1))

        if self.joint_with_seg:
            diff = imagegraph_outputs[2 + 4 * (MAX_DEGREE - 1)] - imagegraph_outputs[2 + 4 * (MAX_DEGREE - 1) + 1]
            # activate
            new_outputs.append(torch.sigmoid(diff))
            new_outputs.append(1.0 - new_outputs[-1])

        # output: [2, 28, 352, 352]
        return torch.cat(new_outputs, dim=1)

    # line 96-99
    def compute_gradients_and_max(self):
        l2loss_grad_max = 0.0

        # Iterate over all parameters to find the max gradient
        for param in self.parameters():
            if param.grad is not None:
                max_grad = torch.max(torch.abs(param.grad))
                if max_grad.item() > l2loss_grad_max:
                    l2loss_grad_max = max_grad.item()
        return l2loss_grad_max

    # 	def Train(self, inputdata, target_prob, target_vector, input_seg_gt, lr):
    # 		feed_dict = {
    # 			self.input_sat : inputdata,
    # 			self.target_prob : target_prob,
    # 			self.target_vector : target_vector,
    # 			self.input_seg_gt : input_seg_gt,
    # 			self.lr : lr,
    # 			self.is_training : True
    # 		}
    #
    # 		ops = [self.loss, self.l2loss_grad_max, self.prob_loss, self.direction_vector_loss, self.seg_loss, self.train_op]
    #
    # 		return self.sess.run(ops, feed_dict=feed_dict)

    def Train(self, inputdata, target_prob, target_vector, input_seg_gt, lr):
        # self.model.train()  # Set the model to training mode
        self.is_training = True
        self.optimizer.zero_grad()  # Zero the gradients
        self.input_seg_gt_target = torch.cat([input_seg_gt + 0.5, 0.5 - input_seg_gt], dim=1)

        output = self.forward(inputdata)

        keypoint_prob_loss, direction_prob_loss, direction_vector_loss, seg_loss = self.SupervisedLoss(
            output, target_prob, target_vector)
        prob_loss = (keypoint_prob_loss + direction_prob_loss)

        # If train_seg, compute the softmax cross-entropy loss between the target segmentation map `input_seg_gt_target` and the model output `linear_output`.
        if self.train_seg:
            # 假设 target 是 one-hot 编码的标签
            seg_target_indices = torch.argmax(self.input_seg_gt_target, dim=1)  # [2, 352, 352]
            loss = nn.CrossEntropyLoss()(output, seg_target_indices)
        else:
            # If not train_seg, compute the softmax cross-entropy loss between the target keypoint_prob map and the model predictions
            if self.joint_with_seg:
                loss = prob_loss + direction_vector_loss + seg_loss
            else:
                loss = prob_loss + direction_vector_loss

        # Compute gradients
        loss.backward()

        # # Compute max gradient value
        self.optimizer.step()  # Update weights
        l2loss_grad_max = self.compute_gradients_and_max()

        return loss.item(), l2loss_grad_max, prob_loss.item(), direction_vector_loss.item(), seg_loss.item()

    # def Evaluate(self, inputdata, target_prob, target_vector, input_seg_gt):
    #     feed_dict = {
    #         self.input_sat: inputdata,
    #         self.target_prob: target_prob,
    #         self.target_vector: target_vector,
    #         self.input_seg_gt: input_seg_gt,
    #         self.is_training: False
    #     }
    #
    #     ops = [self.loss, self.output]
    #
    #     return self.sess.run(ops, feed_dict=feed_dict)

    def Evaluate(self, inputdata, target_prob, target_vector, input_seg_gt):
        # self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient computation
            # Forward pass
            output = self.forward(inputdata)

            # Initialize losses
            keypoint_prob_loss, direction_prob_loss, direction_vector_loss, seg_loss = self.SupervisedLoss(
                output, target_prob, target_vector)

            seg_loss = 0
            prob_loss = keypoint_prob_loss + direction_prob_loss

            if self.train_seg:
                output = nn.Softmax(output)
            else:
                output = self.SoftmaxOutput(output)

            if self.joint_with_seg:
                loss = prob_loss + direction_vector_loss + seg_loss
            else:
                loss = prob_loss + direction_vector_loss

        # Return the total loss and model outputs
        return loss.item(), output

    def saveModel(self, path):
        torch.save(self.state_dict(), path)

    def restoreModel(self, path):
        self.load_state_dict(torch.load(path))

    def addLog(self, test_loss, train_loss, l2_grad):
        # Assuming use of TensorBoard for logging

        if not hasattr(self, 'writer'):
            self.writer = SummaryWriter()
        self.writer.add_scalar('Loss/test', test_loss)
        self.writer.add_scalar('Loss/train', train_loss)
        self.writer.add_scalar('Grad/l2', l2_grad)


class ClassAggregateBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, is_training=True):
        super(ClassAggregateBlock, self).__init__()
        self.is_training = is_training

        # Define the layers inside the block
        self.conv_transpose = ConvLayer(in_ch2, in_ch2, is_training, 3, 3, 2, 2,
                                        batchnorm=True, deconv=True, padding="1")

        self.conv1 = ConvLayer(in_ch1 + in_ch2, in_ch1 + in_ch2, is_training, 3, 3, 1, 1, padding="1")

        self.conv2 = ConvLayer(in_ch1 + in_ch2, out_ch, is_training, 3, 3, 1, 1, padding="1")

    def forward(self, x1, x2):
        x2 = self.conv_transpose(x2)  # Upsample x2

        # use padding, but not sure if it is correct
        if x2.shape[2:] != x1.shape[2:]:
            diffY = x1.size(2) - x2.size(2)
            diffX = x1.size(3) - x2.size(3)
            x2 = F.pad(x2, [0, diffX, 0, diffY])

        x = torch.cat([x1, x2], dim=1)  # Concatenate along the channel dimension
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, is_training=True, kx=3, ky=3,
                 stride_x=2, stride_y=2, batchnorm=False, padding='VALID', add=None, deconv=False, activation='relu',
                 device=None):
        super().__init__()
        self.is_training = is_training
        self.add = add
        self.batchnorm = batchnorm

        if padding == 'VALID':
            self.padding = (0, 0)
        elif padding == 'SAME':
            self.padding = "same"
        elif padding == '1':
            self.padding = (1, 1)
        else:
            raise ValueError(f"Unsupported padding mode: {padding}")

        if not deconv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kx, ky), stride=(stride_x, stride_y),
                                  padding=self.padding, bias=not batchnorm)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(kx, ky), stride=(stride_x, stride_y),
                                           padding=self.padding, bias=not batchnorm)
        if device:
            self.conv = self.conv.to(device)

        self.bn_layer = nn.BatchNorm2d(out_channels) if batchnorm else None

        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        else:
            self.activation_fn = None

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        x = x.to(self.conv.weight.device)
        if self.add is not None:
            if not isinstance(self.add, torch.Tensor):
                raise ValueError("'add' must be a tensor.")
            if self.add.shape != x.shape:
                raise ValueError("The shape of 'add' must match the shape of input 'x'.")
            if self.add.device != x.device:
                self.add = self.add.to(x.device)
            x = x + self.add

        x = self.conv(x)

        if self.batchnorm:
            x = self.bn_layer(x)

        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock, self).__init__()
        self.downsample = downsample

        if downsample:
            self.conv1 = ConvLayer(in_channels, out_channels, kx=3, ky=3, stride_x=2, stride_y=2, padding='1')
        else:
            self.conv1 = ConvLayer(in_channels, out_channels, kx=3, ky=3, stride_x=1, stride_y=1, padding='1')

        self.conv2 = ConvLayer(out_channels, out_channels, kx=3, ky=3, stride_x=1, stride_y=1, padding='1')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if downsample:
            self.downsample_conv = ConvLayer(in_channels, out_channels, kx=1, ky=1, stride_x=2, stride_y=2)
        else:
            self.downsample_conv = ConvLayer(in_channels, out_channels, kx=1, ky=1, stride_x=1, stride_y=1)

    def forward(self, x):
        # [2, 24, 172, 172]
        residual = self.downsample_conv(x) if self.downsample else x

        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)

        return out
