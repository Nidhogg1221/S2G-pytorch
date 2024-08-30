"""
pytorch implementation of the train.py
"""

from model import Sat2GraphModel
from dataloader import Sat2GraphDataLoader as Sat2GraphDataLoaderOSM
from decoder import DecodeAndVis

import numpy as np
import torch.optim as optim
from time import time
import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import argparse
import os
from datetime import datetime
import sys

# import torch.nn as nn


# gpu setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    torch.cuda.set_per_process_memory_fraction(0.95, torch.cuda.current_device())
except AttributeError:
    print("CUDA memory fraction setting is not supported on this system.")

parser = argparse.ArgumentParser()

parser.add_argument('-model_save', action='store', dest='model_save', type=str,
                    help='model save folder ', required=True)
parser.add_argument('-instance_id', action='store', dest='instance_id', type=str,
                    help='instance_id ', required=True)
parser.add_argument('-model_recover', action='store', dest='model_recover', type=str,
                    help='model recover ', required=False, default=None)
parser.add_argument('-image_size', action='store', dest='image_size', type=int,
                    help='instance_id ', required=False, default=256)
parser.add_argument('-lr', action='store', dest='lr', type=float,
                    help='learning rate', required=False, default=0.001)
parser.add_argument('-lr_decay', action='store', dest='lr_decay', type=float,
                    help='learning rate decay', required=False, default=0.5)
parser.add_argument('-lr_decay_step', action='store', dest='lr_decay_step', type=int,
                    help='learning rate decay step', required=False, default=50000)
parser.add_argument('-init_step', action='store', dest='init_step', type=int,
                    help='initial step size ', required=False, default=0)
parser.add_argument('-resnet_step', action='store', dest='resnet_step', type=int,
                    help='instance_id ', required=False, default=8)
parser.add_argument('-spacenet', action='store', dest='spacenet', type=str,
                    help='spacenet folder', required=False, default="")
parser.add_argument('-channel', action='store', dest='channel', type=int,
                    help='channel', required=False, default=12)
parser.add_argument('-mode', type=str, default="train", choices=["train", "test", "validate"],
                    help='Mode: train, test, or validate')
# from train.py, tensorflow implementation
# parser.add_argument('-train_segmentation', action='store', dest='train_segmentation', type=bool,
#                     help='train_segmentation', required =False, default=False)
# parser.add_argument('-spacenet', action='store', dest='spacenet', type=str,
#                     help='spacenet folder', required=False, default="")

args = parser.parse_args()

print(args)

# Setup paths and logging
instance_id = f"{args.instance_id}_{args.image_size}_{args.resnet_step}_channel{args.channel}"
run = f"run-{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}-{instance_id}"
log_folder = os.path.join("alllogs", run)

validation_folder = os.path.join("validation", instance_id)
model_save_folder = os.path.join(args.model_save, instance_id)
os.makedirs(validation_folder, exist_ok=True)
os.makedirs(model_save_folder, exist_ok=True)
print("folder created")
osmdataset = "../data/20cities/"
spacenetdataset = "../data/spacenet/"

batch_size = 2 if args.mode == "train" else 1  # 352 * 352
# batch_size = 4 if args. args.image_size == 384:

max_degree = 6

indrange_train, indrange_test, indrange_validation = [], [], []

for x in range(180):
    if x % 10 < 8:
        indrange_train.append(x)
    elif x % 10 == 9 or x % 20 == 8:
        indrange_test.append(x)
    if x % 20 == 18:
        indrange_validation.append(x)

print("training set", indrange_train)
print("testing set", indrange_test)
print("validation set", indrange_validation)

if args.mode == "train":
    print("train mode start")
    dataloader_train = Sat2GraphDataLoaderOSM(osmdataset, indrange=indrange_train, imgsize=args.image_size,
                                              preload_tiles=4, testing=False, random_mask=True)
    dataloader_train.preload(num=1024)

    print("dataloader_train preloaded")

    dataloader_test = Sat2GraphDataLoaderOSM(osmdataset, indrange=indrange_validation, imgsize=args.image_size,
                                             preload_tiles=len(indrange_validation), testing=True)
    dataloader_test.preload(num=128)

    print("dataloader_test preloaded")
else:
    dataloader = Sat2GraphDataLoaderOSM(osmdataset, indrange=[], imgsize=args.image_size, preload_tiles=1,
                                        random_mask=False, testing=True)
    print("test mode start")
    print("dataloader preloaded")

model = Sat2GraphModel(image_size=args.image_size, resnet_step=args.resnet_step, channel=args.channel,
                       mode=args.mode)
model = model.to(device)

if args.model_recover is not None:
    model.load_state_dict(torch.load(args.model_recover, map_location=device))
else:
    print("Initialize training model")

optimizer = optim.Adam(model.parameters(), lr=args.lr)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
# criterion = nn.MSELoss()  # ? not necessary maybe

writer = SummaryWriter(log_folder)

if args.spacenet == "":
    print("Use the 20-city datasets")


def train():
    while True:
        print("model training method called")
        model.train()

        # getBatch and save the data in memory
        validation_data = []

        test_size = 32

        for j in range(test_size // batch_size):  # 16
            input_sat, gt_prob, gt_vector, gt_seg = dataloader_test.getBatch(batch_size)  # 2
            # reshape the code
            input_sat = torch.tensor(input_sat).to(device)  # From NHWC to NCHW
            if input_sat.shape == torch.Size([2, 352, 352, 3]):
                input_sat = input_sat.permute(0, 3, 1, 2)
            gt_prob = torch.tensor(gt_prob).to(device)  # From NHWC to NCHW
            gt_vector = torch.tensor(gt_vector).to(device)  # From NHWC to NCHW
            gt_seg = torch.tensor(gt_seg).to(device)  # From NHWC to NCHW

            validation_data.append(
                [input_sat.cpu().numpy(), gt_prob.cpu().numpy(), gt_vector.cpu().numpy(), gt_seg.cpu().numpy()])

        # print("getBatch successfully")

        step = args.init_step
        lr = args.lr
        sum_loss = 0
        # reshape
        gt_imagegraph = np.zeros((batch_size, 2 + 4 * max_degree, args.image_size, args.image_size))

        t_load = 0
        t_last = time()
        t_train = 0

        test_loss = 0

        sum_prob_loss = 0
        sum_vector_loss = 0
        sum_seg_loss = 0

        while True:
            t0 = time()
            input_sat, gt_prob, gt_vector, gt_seg = dataloader_train.getBatch(batch_size)  # 2

            input_sat = torch.tensor(input_sat, dtype=torch.float32).to(device)  # From NHWC to NCHW
            if input_sat.shape == torch.Size([2, 352, 352, 3]):
                input_sat = input_sat.permute(0, 3, 1, 2)
            gt_prob = torch.tensor(gt_prob, dtype=torch.float32).to(device)
            gt_vector = torch.tensor(gt_vector, dtype=torch.float32).to(device)
            gt_seg = torch.tensor(gt_seg, dtype=torch.float32).to(device)

            t_load += time() - t0
            #
            t0 = time()

            """model.train called"""
            loss, grad_max, prob_loss, vector_loss, seg_loss = model.Train(input_sat, gt_prob, gt_vector,
                                                                                          gt_seg, lr)
            sum_loss += loss

            sum_prob_loss += prob_loss
            sum_vector_loss += vector_loss
            sum_seg_loss += seg_loss

            t_train += time() - t0
            writer.add_scalar('Time/train_time', t_train, step)
            writer.add_scalar('Time/load_time', t_load, step)


            if step % 10 == 0 and step > 0:
                sys.stdout.write("\rbatch:%d " % step + ">>" * ((step - (step // 200) * 200) // 10) + "--" * (
                        ((step // 200 + 1) * 200 - step) // 10))
                sys.stdout.flush()

            # calculate the loss and save the data every 200 steps
            if step > -1 and step % 200 == 0:
                sum_loss //= 200

                writer.add_scalar('Loss/train', sum_loss, step)
                writer.add_scalar('Loss/test', test_loss, step)
                writer.add_scalar('Loss/prob_loss', sum_prob_loss / 200.0, step)
                writer.add_scalar('Loss/vector_loss', sum_vector_loss / 200.0, step)
                writer.add_scalar('Loss/seg_loss', sum_seg_loss / 200.0, step)

                # calculate the test loss every 1000 steps
                if step % 1000 == 0 or (step < 1000 and step % 200 == 0):

                    for j in range(-1, test_size // batch_size):  # 32 // 2 = 16
                        # validation data already divided in line 148, no need to modify
                        if j >= 0:
                            input_sat, gt_prob, gt_vector, gt_seg = validation_data[j][0], validation_data[j][1], \
                                validation_data[j][2], validation_data[j][3]
                        if j == 0:
                            test_loss = 0
                            test_gan_loss = 0

                        # there must be a more elegent way to handle this
                        if isinstance(gt_imagegraph, np.ndarray):
                            gt_imagegraph = torch.tensor(gt_imagegraph).to('cuda')

                        # if gt_prob is a numpy array, convert it to a PyTorch tensor and make sure it is on the GPU
                        if isinstance(gt_prob, np.ndarray):
                            gt_prob = torch.tensor(gt_prob).to('cuda')
                        else:
                            gt_prob = gt_prob.to('cuda')

                        if isinstance(gt_vector, np.ndarray):
                            gt_vector = torch.tensor(gt_vector).to('cuda')
                        else:
                            gt_vector = gt_vector.to('cuda')

                        gt_imagegraph[:, 0:2, :, :] = gt_prob[:, 0:2, :, :]
                        gt_imagegraph = torch.tensor(gt_imagegraph).to('cuda')

                        for k in range(max_degree):
                            gt_imagegraph[:, 2 + k * 4:2 + k * 4 + 2, :, :] = gt_prob[:, 2 + k * 2:2 + k * 2 + 2, :, :]
                            gt_imagegraph[:, 2 + k * 4 + 2:2 + k * 4 + 4, :, :] = gt_vector[:, k * 2:k * 2 + 2, :, :]

                        # _ stands for temporary/internal variable
                        input_sat = torch.tensor(input_sat, dtype=torch.float32).to(device)  # From NHWC to NCHW
                        if input_sat.shape == torch.Size([2, 352, 352, 3]):
                            input_sat = input_sat.permute(0, 3, 1, 2)
                        gt_prob = torch.tensor(gt_prob, dtype=torch.float32).to(device)
                        gt_vector = torch.tensor(gt_vector, dtype=torch.float32).to(device)
                        gt_seg = torch.tensor(gt_seg, dtype=torch.float32).to(device)

                        # evaluate called 16times every 200 steps
                        _test_loss, output = model.Evaluate(input_sat, gt_prob, gt_vector, gt_seg)
                        test_loss += _test_loss

                        if step == 1000 or step % 2000 == 0 or (step < 1000 and step % 200 == 0):
                            for k in range(batch_size):  # 2

                                # input_sat_img shape: torch.Size([2, 3, 352, 352])
                                input_sat_img = ((input_sat[k] + 0.5) * 255.0).permute(1, 2, 0).cpu().numpy().astype(
                                    np.uint8)  # (352, 352, 3)
                                writer.add_image(f'Input_Satellite/tile_{k}', input_sat_img, step, dataformats='HWC')

                                # segmentation output (joint training)
                                # output shape:  torch.Size([2, 28, 352, 352])
                                output_img = (output[k, -2, :, :] * 255.0).cpu().numpy().reshape(
                                    (args.image_size, args.image_size)).astype(np.uint8)

                                # output_img shape:  (352, 352)
                                writer.add_image(f'Output_Segmentation/tile_{k}', output_img[:, :, np.newaxis], step,
                                                 dataformats='HWC')
                                Image.fromarray(output_img).save(
                                    f"{validation_folder}/tile{j * batch_size + k}_output_seg.png")

                                # print("gt_seg_numpy shape: ", gt_seg_numpy.shape)  (352, 352)
                                gt_seg_numpy = ((gt_seg[k, 0, :, :].cpu().numpy() + 0.5) * 255.0).reshape(
                                    (args.image_size, args.image_size)).astype(np.uint8)
                                Image.fromarray(gt_seg_numpy).save(
                                    validation_folder + "/tile%d_gt_seg.png" % (j * batch_size + k))

                                # keypoints
                                # print("output_keypoints_img shape: ", output_keypoints_img.shape)  (352, 352)
                                output_keypoints_img = (output[k, 0, :, :].cpu().numpy() * 255.0).astype(np.uint8)
                                Image.fromarray(output_keypoints_img).save(
                                    validation_folder + "/tile%d_output_keypoints.png" % (j * batch_size + k))

                                # input satellite  (352, 352, 3)
                                Image.fromarray(input_sat_img).save(
                                    validation_folder + "/tile%d_input_sat.png" % (j * batch_size + k))
                                DecodeAndVis(output[k, 0:2 + 4 * max_degree, :, :].reshape(
                                    (args.image_size, args.image_size, 2 + 4 * max_degree)),
                                    validation_folder + "/tile%d_output_graph_0.01_snap.png" % (
                                            j * batch_size + k), thr=0.01, snap=True, imagesize=args.image_size)

                    test_loss /= test_size // batch_size  # 16

                print("")
                print("step", step, "loss", sum_loss, "test_loss", test_loss, "prob_loss", sum_prob_loss / 200.0,
                      "vector_loss", sum_vector_loss / 200.0, "seg_loss", sum_seg_loss / 200.0)

                sum_prob_loss = 0
                sum_vector_loss = 0
                sum_seg_loss = 0
                sum_loss = 0

            if step > 0 and step % 400 == 0:
                dataloader_train.preload(num=1024)

            if step > 0 and step % 2000 == 0:
                print(time() - t_last, t_load, t_train)
                t_last = time()
                t_load = 0
                t_train = 0

            if step > 0 and (step % 10000 == 0):
                model.saveModel(model_save_folder + "model%d" % step)

            # adjust the learning rate according to the predetermined steps to slow down the learning speed in the
            # later stage of training, so as to adjust the model weights more finely.
            if step > 0 and step % args.lr_decay_step == 0:
                lr = lr * args.lr_decay
                writer.add_scalar('Learning_Rate', lr, step)

            step += 1
            if step == 300000 + 2:
                # """调试用"""
                # if step == 300+2:
                break


def evaluate():
    model.eval()
    with torch.no_grad():
        for tile_id in (indrange_test if args.mode == "test" else indrange_validation):
            t0 = time()
            """
            tf: [batch, height, width, channel]
            pytorch: [batch, channel, height, width]
            """
            input_sat, gt_prob, gt_vector = dataloader.loadtile(tile_id)

            gt_seg = np.zeros((1, 1, args.image_size, args.image_size))

            input_sat = torch.tensor(input_sat, dtype=torch.float32).to(device)  # From NHWC to NCHW
            if input_sat.shape == torch.Size([2, 352, 352, 3]):
                input_sat = input_sat.permute(0, 3, 1, 2)
            gt_prob = torch.tensor(gt_prob, dtype=torch.float32).to(device)
            gt_vector = torch.tensor(gt_vector, dtype=torch.float32).to(device)
            gt_seg = torch.tensor(gt_seg, dtype=torch.float32).to(device)

            """ 
            transform numpy array to tensor
            map ftn apply the function to each element in the list
            lambda transfer the input to tensor, and move the tensor to device
            """
            input_sat, gt_prob, gt_vector = map(lambda x: torch.tensor(x).to(device), [input_sat, gt_prob, gt_vector])
            # gt_imagegraph = np.zeros((26, 2048, 2048))

            gt_imagegraph[:, 0:2, :, :] = gt_prob[:, 0:2, :, :].cpu().numpy()
            gt_imagegraph = torch.tensor(gt_imagegraph).to(device)

            for k in range(max_degree):
                gt_imagegraph[2 + k * 4:2 + k * 4 + 2, :, :] = gt_prob[0, 2 + k * 2:2 + k * 2 + 2, :, :]
                gt_imagegraph[2 + k * 4 + 2:2 + k * 4 + 4, :, :] = gt_vector[0, k * 2:k * 2 + 2, :, :]

            x, y = 0, 0

            output = np.zeros((2 + 4 * 6 + 2, 2048 + 64, 2048 + 64))

            mask = np.ones((2048 + 64, 2048 + 64, 2 + 4 * 6 + 2)) * 0.001
            weights = np.ones((2 + 4 * 6 + 2, args.image_size, args.image_size)) * 0.001
            weights[:, 32: args.image_size - 32, 32: args.image_size - 32] = 0.5
            weights[:, 56: args.image_size - 56, 56: args.image_size - 56] = 1.0
            weights[:, 88: args.image_size - 88, 88: args.image_size - 88] = 1.5

            input_sat = np.pad(input_sat, ((0, 0), (0, 0), (32, 32), (32, 32)), 'constant')
            gt_vector = np.pad(gt_vector, ((0, 0), (0, 0), (32, 32), (32, 32)), 'constant')
            gt_prob = np.pad(gt_prob, ((0, 0), (0, 0), (32, 32), (32, 32)), 'constant')

            # Process the image in patches
            # traverse the image with sliding window
            for x in range(0, 352 * 6 - 176 - 88, 176 // 2):
                progress = x / 88
                sys.stdout.write(
                    f"\rProcessing Tile {tile_id} ...  " + ">>" * int(progress) + "--" * (20 - int(progress)))
                sys.stdout.flush()

                for y in range(0, 352 * 6 - 176 - 88, 176 // 2):
                    input_slice = torch.tensor(input_sat[:, x:x + args.image_size, y:y + args.image_size, :],
                                               dtype=torch.float32).to(device)

                    gt_prob_slice = torch.tensor(gt_prob[:, x:x + args.image_size, y:y + args.image_size, :],
                                                 dtype=torch.float32).to(device)
                    gt_vector_slice = torch.tensor(gt_vector[:, x:x + args.image_size, y:y + args.image_size, :],
                                                   dtype=torch.float32).to(device)
                    gt_seg_slice = torch.tensor(gt_seg, dtype=torch.float32).to(device)

                    alloutputs = model.Evaluate(input_slice, gt_prob_slice, gt_vector_slice, gt_seg_slice)
                    _output = alloutputs[-1]

                    mask[x:x + args.image_size, y:y + args.image_size, :] += weights
                    output[x:x + args.image_size, y:y + args.image_size, :] += np.multiply(_output[0, :, :, :],
                                                                                           weights)

            # Normalize the output by the mask

            output = np.divide(output, mask)
            output = torch.tensor(output).cpu().numpy()

            # Crop the padding off
            output = output[:, 32:2048 + 32, 32:2048 + 32]
            input_sat = input_sat[:, :, 32:2048 + 32, 32:2048 + 32]

            # Save the output keypoints image
            output_keypoints_img = (output[0, :, :] * 255.0).reshape((2048, 2048)).astype(np.uint8)
            Image.fromarray(output_keypoints_img).save(f"outputs/region_{tile_id}_output_keypoints.png")

            # Save the input satellite image
            input_sat_img = ((input_sat[0, :, :, :] + 0.5) * 255.0).reshape((3, 2048, 2048)).astype(np.uint8)
            Image.fromarray(input_sat_img).save(f"outputs/region_{tile_id}_input.png")

            # Decode and visualize the output
            DecodeAndVis(output, f"outputs/region_{tile_id}_output", thr=0.05, edge_thr=0.05, snap=True, imagesize=2048)

            # Save the raw output
            np.save(f"rawoutputs_{args.instance_id}/region_{tile_id}_output_raw", output)

            print(f" done!  time: {time() - t0:.2f} seconds")


# Run the appropriate mode
if args.mode == "train":
    train()
elif args.mode in ["validate", "test"]:
    evaluate()
else:
    raise ValueError("Unsupported mode! Choose between 'train', 'validate', or 'test'.")
