import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume, read_csv, save_two_predictions_and_masks_randomly, get_session_name

def validate_one_epoch(session_name, args, val_loader, model, optimizer,writer, epoch_num, img_save_path_val):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_epochs * len(trainloader)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # one epoch
    dice_sum = 0
    loss_sum = 0
    for i_batch, sampled_batch in enumerate(val_loader):
        image_batch, label_batch, text_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
        image_batch, label_batch, text_batch = image_batch.cuda(), label_batch.cuda(), text_batch.cuda()
        outputs = model(image_batch,text_batch)
        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch, softmax=True)
        loss = 0.4 * loss_ce + 0.6 * loss_dice

        ## Not sure if these are okay
        dice_sum += 1-dice_loss
        loss_sum += 1-loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/total_loss', loss, iter_num)
        writer.add_scalar('info/loss_ce', loss_ce, iter_num)

        logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

        if iter_num % 20 == 0:
            image = image_batch[1, 0:1, :, :]
            image = (image - image.min()) / (image.max() - image.min())
            writer.add_image('val/Image', image, iter_num)
            outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            writer.add_image('val/Prediction', outputs[1, ...] * 50, iter_num)
            labs = label_batch[1, ...].unsqueeze(0) * 50
            writer.add_image('val/GroundTruth', labs, iter_num)
    save_two_predictions_and_masks_randomly(outputs, label_batch, f"{session_name}_{epoch_num}", img_save_path_val)
    return loss_sum/len(val_loader), dice_sum/len(val_loader)

def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    # logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)

def trainer_synapse(args, model, snapshot_path):
    session_name = get_session_name()
    if not os.path.exists('./training_output'):
        os.mkdir('./training_output')
    img_save_path = './training_output/' + session_name
    img_save_path_train = img_save_path + '/train'
    img_save_path_val = img_save_path + '/val'
    os.mkdir(img_save_path)
    os.mkdir(img_save_path_train)
    os.mkdir(img_save_path_val)
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    train_text = read_csv(args.root_path + '/synapse_train.csv')
    val_text = read_csv(args.root_path + '/synapse_val.csv')
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, row_text=train_text, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))
    # needs val.txt at list_dir
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, row_text=val_text, split="val",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    max_dice = 0
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch, text_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
            image_batch, label_batch, text_batch = image_batch.cuda(), label_batch.cuda(), text_batch.cuda()
            outputs = model(image_batch,text_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_two_predictions_and_masks_randomly(outputs, label_batch, f"{session_name}_{epoch_num}", img_save_path_train)

        # validation 
        val_optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = validate_one_epoch(session_name, args, val_loader,
                                                    model,
                                                 val_optimizer, writer, epoch_num, img_save_path_val)
            
        #### Save best model
        if val_dice > max_dice:
            if epoch_num + 1 > 5:
                # logger.info(
                #     '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch_num + 1
                save_checkpoint({'epoch': epoch_num,
                                 'best_model': True,
                                #  'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                #  'optimizer': optimizer.state_dict()
                                'best_epoch': best_epoch,
                                }, 
                                 f"./saved_best/model/{session_name}_{epoch_num}")



        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"