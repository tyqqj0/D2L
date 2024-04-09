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
from torch.cuda.amp import GradScaler

import utils.arg.parser
from epochs import TrainEpoch, ValEpoch, LIDComputeEpoch, NEComputeEpoch, ExpressionSaveEpoch, PCACorrectEpoch, \
    BaseEpoch, plot_kmp, ClusterBackwardEpoch

from loss import lid_paced_loss
from model.resnet18 import ResNet18FeatureExtractor
from model.resnet50 import ResNet50FeatureExtractor
from utils.BOX import logbox
from utils.data import load_data
from utils.plotfn import plot_layers_seaborn, dict_to_json
from utils.text import text_in_box

plot_layer_all = logbox.log_artifact_autott(plot_layers_seaborn)
dict_to_json = logbox.log_artifact_autott(dict_to_json)

# os.environ['OMP_NUM_THREADS'] = '1'


# logbox = box()


def train(model, train_loader, test_loader, optimizer, criterion, scheduler, device, args, logbox):
    # 实例化epoch
    # 通过BaseEpoch设置最大epoch数
    stop_count = 0
    timert = utils.EpochTimer()
    BaseEpoch.set_max_epoch(args.epochs)
    # cluster_model = KMeans(n_clusters=args.num_classes)
    train_epoch = TrainEpoch(model, train_loader, optimizer, criterion, device,
                             scaler=GradScaler() if args.amp else None)
    val_epoch = ValEpoch(model, test_loader, criterion, device, plot_wrong=args.plot_wrong,
                         replace_label=(args.dataset != 'MNIST'))
    lid_compute_epoch = LIDComputeEpoch(model, train_loader, device, num_class=args.num_classes,
                                        group_size=args.knowledge_group_size, interval=args.plot_interval)
    expression_save_epoch = ExpressionSaveEpoch(model, train_loader, device, args.expression_data_loc,
                                                f'{args.model}_{args.noise_ratio}', times=args.expression_data_time,
                                                num_class=args.num_classes, group_size=args.knowledge_group_size,
                                                interval=args.plot_interval)
    ne_compute_epoch = NEComputeEpoch(model, train_loader, device, num_class=args.num_classes,
                                      group_size=args.knowledge_group_size, interval=args.plot_interval, bar=False)
    pca_compute_epoch = PCACorrectEpoch(model, train_loader, device, num_class=args.num_classes,
                                        group_size=args.knowledge_group_size, interval=args.plot_interval, bar=False)
    cluster_backward_epoch = ClusterBackwardEpoch(model, train_loader, args.cluster_model, device,
                                                  num_class=args.num_classes,
                                                  group_size=args.knowledge_group_size, interval=args.plot_interval, bar=False)#

    for epoch in range(args.epochs):
        timert._start()

        if stop_count > 5 and epoch < args.epochs - 1:
            continue

        print('\n')
        print(text_in_box('Epoch: %d/%d' % (epoch + 1, args.epochs)))
        ne_dict = None
        train_loss, train_accuracy = train_epoch.run(epoch + 1)
        val_loss, val_accuracy = val_epoch.run(epoch + 1)
        # knowes, logits_list = lid_compute_epoch.run(epoch + 1)
        # expression_save_epoch.run(epoch + 1, val_accuracy=val_accuracy)
        # ne_dict = ne_compute_epoch.run(epoch + 1)
        # pca_compute_epoch.run(epoch + 1)
        cluster_backward_epoch.run(epoch + 1)

        # if args.lossfn == 'l2d' or args.lossfn == 'lid_paced_loss':
        #     criterion.update(knowes, epoch + 1)
        scheduler.step()

        # 打印训练信息
        print('\n')
        print('train_loss: %.3f, train_accuracy: %.3f' % (train_loss, train_accuracy))  # , train_lid
        print('val_loss: %.3f, val_accuracy: %.3f' % (val_loss, val_accuracy))  # , val_lid

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

        # mlflow记录图像
        if ((epoch + 1) % args.plot_interval == 0 or epoch + 1 == args.epochs) and args.plot_interval != -1:
            # print('knowledge:', knowes)
            # print('ne:', ne_dict)
            # logbox.log_metrics('knowledge', knowes, step=epoch + 1)
            logbox.log_metrics('ne', ne_dict, step=epoch + 1)

            # 绘制knows图像
            # plot_layer_all(knowes, epoch + 1, y_lim=25, folder='knowledge',
            #                pre=args.model + '_' + str(args.noise_ratio))

            # 绘制ne图像
            # plot_layer_all(ne_dict, epoch + 1, y_lim=6, folder='ne', pre=args.model + '_' + str(args.noise_ratio))

            # 保存knows参数文件数
            # dict_to_json(knowes.update(
            #     {'info:model': args.model, 'info:noise_ratio': args.noise_ratio, 'info:data_set': args.dataset}),
            #     epoch + 1, pre=args.model + '_' + str(int(args.noise_ratio * 100)))

            # 绘制知识图谱, 遍历每个类，传入当前epoch的层logits
            # if args.knowledge_group_size > 1:
            #     plot_kmp(epoch, logits_list, model_name=args.model, noise_ratio=args.noise_ratio, folder='kn_map')

        # 提前停止条件, 若多个epoch训练准确率都在1左右
        if train_accuracy >= 0.95:
            print('near cpt', stop_count)
            stop_count += 1

        else:
            stop_count = 0

        # MLflow记录模型
        if ((epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs) and args.save_interval != -1:
            logbox.save_model(model.state_dict())

        # 打印记录时间
        print(timert)
        t, speed_pm, speed_ph = timert._end()
        logbox.log_metrics('time', {'time': t, 'speed_pm': speed_pm, 'speed_ph': speed_ph}, step=epoch + 1)

    # MLflow记录参数
    logbox.log_params({
        'lr_finel': scheduler.get_last_lr()[0],
    })  # 将args转换为字典


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


if __name__ == '__main__':
    # args

    # utils.arg.parser.save_parser_to_json(parser)
    main()
