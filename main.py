# -*- CODING: UTF-8 -*-
# @time 2024/1/24 19:39
# @Author tyqqj
# @File main.py
# @
# @Aim

import os

# import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import utils.arg.parser
from LID import get_lids_batches
from model.resnet18 import ResNet18FeatureExtractor
from utils.BOX.box2 import box
from utils.data import load_data
from utils.text import text_in_box

logbox = box()
plot_lif_all = None


def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    logits_list = None
    # print('\n')
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, (inputs, targets) in progress_bar:
        # print('start')
        # print(inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, logits = model(inputs)
        if logits_list is None:
            logits_list = {key: [] for key in logits.keys()}
        for key, value in logits.items():
            logits_list[key].append(value)
        loss = criterion(outputs, targets)
        # print(f"{batch_idx}: loss:", loss.item())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # 更新进度条显示的信息
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
        # print('end')
    # print('\n')
    train_loss = running_loss / len(data_loader)
    train_accuracy = correct / total
    lidss = get_lids_batches(logits_list)
    return train_loss, train_accuracy, lidss


def val_epoch(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    logits_list = None
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, logits = model(inputs)
            if logits_list is None:
                logits_list = {key: [] for key in logits.keys()}
            for key, value in logits.items():
                logits_list[key].append(value)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    val_loss = running_loss / len(data_loader)
    val_accuracy = correct / total
    lidss = get_lids_batches(logits_list)
    return val_loss, val_accuracy, lidss


def train(model, train_loader, test_loader, optimizer, criterion, scheduler, device, args, logbox):
    for epoch in range(args.epochs):
        print('\n')
        print(text_in_box('Epoch: %d/%d' % (epoch + 1, args.epochs)))
        train_loss, train_accuracy, train_lid = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy, val_lid = val_epoch(model, test_loader, criterion, device)

        scheduler.step()

        # 打印训练信息

        print('train_loss: %.3f, train_accuracy: %.3f, train_lid:' % (train_loss, train_accuracy))#, train_lid
        print('val_loss: %.3f, val_accuracy: %.3f, val_lid:' % (val_loss, val_accuracy))#, val_lid

        # mlflow记录
        train_matrics = {
            'loss': train_loss,
            'accuracy': train_accuracy,
        }
        val_matrics = {
            'loss': val_loss,
            'accuracy': val_accuracy,
        }
        logbox.log_metrics('train', train_matrics, step=epoch + 1)
        logbox.log_metrics('val', val_matrics, step=epoch + 1)
        logbox.log_metrics('train', train_lid[0], pre='lid', step=epoch + 1)
        logbox.log_metrics('train', train_lid[1], pre='Dim_pr', step=epoch + 1)
        logbox.log_metrics('val', val_lid[1], pre='Dim_pr', step=epoch + 1)
        logbox.log_metrics('val', val_lid[0], pre='lid', step=epoch + 1)
        # mlflow记录图像
        if ((epoch + 1) % args.plot_interval == 0 or epoch + 1 == args.epochs) and args.plot_interval != -1:
            plot_lid_all(train_lid[0], epoch + 1, y_lim=25, folder='train_lid/', pre='lid')
            plot_lid_all(train_lid[1], epoch + 1, y_lim=0.025, folder='train_lid_pr/', pre='lid_pr')

    # MLflow记录参数
    logbox.log_params({
        'epoch': epoch,
        'lr': scheduler.get_last_lr()[0],
        'momentum': args.momentum,
        'weight_decay': args.weight_decay
    })
    # MLflow记录模型
    if ((epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs) and args.save_interval != -1:
        logbox.save_model(model.state_dict())


def main(args):
    # 设置mlflow
    # mlflow.set_tracking_uri("http://localhost:5002")
    global logbox

    logbox.set_dataset_name(dataset_name=args.dataset)
    logbox.set_model_name(model_name=args.model)
    logbox.set_optional_info(str(args.noise_ratio))
    # 获取数据集
    train_loader, test_loader, args.num_classes, args.in_channels = load_data(path='D:/gkw/data/classification',
                                                                              dataset_name=args.dataset,
                                                                              max_data=args.max_data,
                                                                              batch_size=args.batch_size,
                                                                              noise_ratio=args.noise_ratio,
                                                                              noise_type=args.noise_type)
    # if torch.cuda.is_available():
    #     train_loader = train_loader.cuda()
    #     test_loader = test_loader.cuda()
    # print("train_loader:", train_loader)
    # print("test_loader:", test_loader)
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # 设置模型
    if args.model == 'resnet18':
        model = ResNet18FeatureExtractor(pretrained=False, num_classes=args.num_classes, in_channels=args.in_channels)
        if torch.cuda.is_available():
            model = model.cuda()
    else:
        raise NotImplementedError('model not implemented!')

    # 设置优化器
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    # 设置学习率调整策略
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # mlflow.start_run(run_name=args.run_name)
    # 训练
    with logbox:
        train(model, train_loader, test_loader, optimizer, criterion, scheduler, device, args, logbox)
    # mlflow.end_run()


@logbox.log_artifact_autott
def plot_lid_all(lidss, epoch, y_lim=None, folder='', pre='', path=None):
    file_name = folder + pre + 'epoch_{}.png'.format(epoch)
    # 如果文件夹不存在
    if not os.path.exists(path + folder):
        os.makedirs(path + folder)
    import matplotlib.pyplot as plt
    plt.figure()
    layers = list(lidss.keys())
    values = [lidss[layer] for layer in layers]
    plt.bar(layers, values)
    if y_lim:
        plt.ylim((0, y_lim))
    plt.xlabel('Layers')
    plt.ylabel('Values')
    plt.title(f'Layer Values of {folder} at Epoch {epoch}')
    # plt.legend()
    plt.savefig(path + file_name)
    print('save plot {}'.format(file_name))
    plt.close()
    return path + file_name


if __name__ == '__main__':
    # args
    args = utils.arg.parser.get_args('run.json')
    # utils.arg.parser.save_parser_to_json(parser)
    main(args)
