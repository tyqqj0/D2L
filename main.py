# -*- CODING: UTF-8 -*-
# @time 2024/1/24 19:39
# @Author tyqqj
# @File main.py
# @
# @Aim

import os
from collections import defaultdict

# import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import utils.arg.parser
from LID import get_lids_batches
from loss import lid_paced_loss
from model.resnet18 import ResNet18FeatureExtractor
from model.resnet50 import ResNet50FeatureExtractor
from utils.BOX.box2 import box
from utils.data import load_data
from utils.plotfn import plot_lid_seaborn, kn_map, plot_wrong_label
from utils.text import text_in_box

logbox = box()
plot_lid_all = logbox.log_artifact_autott(plot_lid_seaborn)
plot_kn_map = logbox.log_artifact_autott(kn_map)
plot_wrong_label = logbox.log_artifact_autott(plot_wrong_label)


def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # print('\n')
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, (inputs, targets) in progress_bar:
        # print('start')
        # print(inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, _ = model(inputs)

        # print('outputs:', outputs.shape, 'targets:', targets.shape)
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

    return train_loss, train_accuracy


def val_epoch(model, data_loader, criterion, device, plot_wrong, epoch=0):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, _ = model(inputs)

            loss = criterion(outputs, targets)

            if batch_idx == 0 and plot_wrong > 0:
                _, predicted = torch.max(outputs.data, 1)
                plot_wrong_label(inputs, targets, predicted, epoch, folder='wrong_output', max_samples=plot_wrong)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    val_loss = running_loss / len(data_loader)
    val_accuracy = correct / total
    return val_loss, val_accuracy


def lid_compute_epoch(model, data_loader, device, num_class=10, group_size=15, epoch=0, model_name='resnet18'):
    '''
    计算LID
    :param model:
    :param data_loader: 使用训练集
    :param device:
    :param group_size: 计算LID时的每类取数据量, 在分类任务时现阶段使用，后序将计算Y密度/X密度代替
    '''
    if group_size < 2:
        return {'null': 0}
    model.eval()
    logits_list = defaultdict(dict)
    # 存储每个类别的logits,defaultdict是一个字典，当字典里的key不存在但被查找时，返回的不是keyEror而是一个默认值
    class_counts = [0] * num_class  # 记录每个类别收集的样本数

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # 如果targets不是一维的，就转换成一维的softmax
            if len(targets.size()) > 1:
                targets = torch.argmax(targets, dim=1)

            # optimizer.zero_grad()
            outputs, logits = model(inputs)  # 假设模型返回的最后一个元素是logits

            # 遍历batch中的每个样本
            for idx, target in enumerate(targets):
                label = target.item()
                # 检查是否已经有足够的样本
                if class_counts[label] < group_size:
                    # 此处logits_list[label]应该是一个字典,结构为{'layer_name', data_tensor}应在data_tensor处拼接新的logits
                    for key, value in logits.items():
                        # logits_list
                        if key in logits_list[label]:
                            logits_list[label][key] = torch.cat((logits_list[label][key], value[idx].unsqueeze(0)),
                                                                dim=0)
                        else:
                            logits_list[label][key] = value[idx].unsqueeze(0)
                    class_counts[label] += 1
                # 如果每个类别都收集到了足够的样本，就退出
                if all(count >= group_size for count in class_counts):
                    break
            # 如果每个类别都收集到了足够的样本，就退出
            if all(count >= group_size for count in class_counts):
                break

    # 计算每个类别的LID
    class_lidses = []
    for label, logits_per_class in logits_list.items():
        # 假设get_lids_batches是计算LID的函数
        # 这里需要将logits转换为tensor，因为它目前是一个列表
        # tensor_logits = torch.stack(logits_per_class)
        # print(logits_per_class.shape)
        class_lidses.append(get_lids_batches(logits_per_class))

    # 求class_lidses的平均值
    lidses = {key: 0 for key in class_lidses[0].keys()}
    for a_lids in class_lidses:
        for key, value in a_lids.items():
            lidses[key] += value
    for key in lidses.keys():
        lidses[key] = lidses[key] / len(class_lidses)

    # # 绘制知识图谱, 遍历每个类，传入当前epoch的层logits
    # for label, logits_per_class in logits_list.items():
    #     # a_class_layer = logits_per_class.keys()
    #     if label != 6:
    #         continue
    #     plot_kn_map(logits_per_class, label, epoch=epoch, group_size=group_size, folder='kn_map',
    #                 pre=model_name)

    #
    return lidses


def train(model, train_loader, test_loader, optimizer, criterion, scheduler, device, args, logbox):
    for epoch in range(args.epochs):
        print('\n')
        print(text_in_box('Epoch: %d/%d' % (epoch + 1, args.epochs)))
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = val_epoch(model, test_loader, criterion, device, plot_wrong=args.plot_wrong, epoch=epoch + 1)
        knowes = lid_compute_epoch(model, train_loader, device, num_class=args.num_classes,
                                   group_size=args.knowledge_group_size, epoch=epoch, model_name=args.model)
        if args.lossfn == 'l2d' or args.lossfn == 'lid_paced_loss':
            criterion.update(knowes, epoch + 1)
        scheduler.step()

        # 打印训练信息

        print('train_loss: %.3f, train_accuracy: %.3f' % (train_loss, train_accuracy))  # , train_lid
        print('val_loss: %.3f, val_accuracy: %.3f' % (val_loss, val_accuracy))  # , val_lid
        print('knowledge:', knowes)

        # mlflow记录
        train_metrics = {
            'loss': train_loss,
            'accuracy': train_accuracy,
        }
        val_metrics = {
            'loss': val_loss,
            'accuracy': val_accuracy,
        }
        logbox.log_metrics('train', train_metrics, step=epoch + 1)
        logbox.log_metrics('val', val_metrics, step=epoch + 1)
        logbox.log_metrics('knowledge', knowes, step=epoch + 1)
        # mlflow记录图像
        if ((epoch + 1) % args.plot_interval == 0 or epoch + 1 == args.epochs) and args.plot_interval != -1:
            plot_lid_all(knowes, epoch + 1, y_lim=25, folder='knowledge', pre=args.model + '_' + str(args.noise_ratio))
            dict_to_json(knowes.update(
                {'info:model': args.model, 'info:noise_ratio': args.noise_ratio, 'info:data_set': args.dataset}),
                epoch + 1, pre=args.model + '_' + str(int(args.noise_ratio * 100)))

        # MLflow记录模型
        if ((epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs) and args.save_interval != -1:
            logbox.save_model(model.state_dict())

    # MLflow记录参数
    logbox.log_params({
        'lr': scheduler.get_last_lr()[0],
    }.update(vars(args))) # 将args转换为字典

def main():
    # 设置mlflow
    # mlflow.set_tracking_uri("http://localhost:5002")
    args = utils.arg.parser.get_args('run.json')
    global logbox

    logbox.set_dataset_name(dataset_name=args.dataset)
    logbox.set_model_name(model_name=args.model)
    logbox.set_optional_info(str(args.noise_ratio) + '_' + str(args.lossfn))
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
    elif args.model == 'resnet50':
        model = ResNet50FeatureExtractor(pretrained=False, num_classes=args.num_classes, in_channels=args.in_channels)
        if torch.cuda.is_available():
            model = model.cuda()
    else:
        raise NotImplementedError('model not implemented!')

    # 设置优化器
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # 设置损失函数
    if args.lossfn == 'ce' or args.lossfn == 'cross_entropy':
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss()
    elif args.lossfn == 'l2d' or args.lossfn == 'lid_paced_loss':
        # raise NotImplementedError('lid loss 还未实现！')
        criterion = lid_paced_loss(max_epochs=args.epochs, beta1=0.1, beta2=1.0)
    else:
        raise NotImplementedError('loss function not implemented!')
    # 设置学习率调整策略
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # mlflow.start_run(run_name=args.run_name)
    # 训练
    with logbox:
        train(model, train_loader, test_loader, optimizer, criterion, scheduler, device, args, logbox)
    # mlflow.end_run()


# 将knowledge的字典转换为json并保存提交
@logbox.log_artifact_autott
def dict_to_json(dicttt, epoch, folder='knowledge_json', pre='', path=''):
    import json
    import os
    file_name = pre + '_' + 'epoch_{:03d}.json'.format(epoch)
    # 如果path不为None，则在path中创建文件夹
    full_folder_path = os.path.join(path, folder) if path is not None else folder
    if not os.path.exists(full_folder_path):
        os.makedirs(full_folder_path)
    full_file_path = full_folder_path + '/'
    full_file_path = full_file_path + file_name

    with open(full_file_path, 'w') as f:
        # 处理格式为自动缩进

        json.dump(dicttt, f)
    return full_folder_path


if __name__ == '__main__':
    # args

    # utils.arg.parser.save_parser_to_json(parser)
    main()
