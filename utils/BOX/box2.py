# -*- CODING: UTF-8 -*-
# @time 2024/1/25 12:49
# @Author tyqqj
# @File box2.py
# @
# @Aim
import functools

import mlflow
import time

import utils.arg.parser as parser
import utils.text as text


def ensure_running(func):
    def wrapper(self, *args, **kwargs):
        if not self.running:
            print(f"Function {func.__name__} cannot be called because the 'running' flag is False.")
            return None
        return func(self, *args, **kwargs)

    return wrapper


class box:
    def __init__(self, arg_path='run.json'):
        self.arg = parser.get_args(arg_path)
        # mlflow相关参数
        # 实验名称建议格式: 任务名称(手动指定)_模型名称(半自动)_数据集名称(半自动)_可选信息列表(如噪声率)(半自动)_时间(自动)
        self.exp_name = self.arg.exp_name
        # self.run_name = self.arg.run_name
        self.mission_name = self.arg.run_name
        self.model_name = ''
        self.dataset_name = ''
        self.optional_info = ''
        self.time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.tracking_uri = './mlruns'

        # 运行状态标记
        self.running = False
        self.local_run_id = None
        self.epoch = 0  # 可选

        # 工件暂存位置
        self.cache_dir = self.arg.cache_dir

        # 显示初始化参数信息
        print(text.text_in_box(f'Init box with {arg_path}'))
        print('\texp_name:', self.exp_name)
        print('\trun_name:', self.run_name)
        print('\ttracking_uri: file:///', self.tracking_uri)
        print(text.split_line())

    def set_model_name(self, model_name):
        self.model_name = model_name

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def set_optional_info(self, optional_info):
        if isinstance(optional_info, list):
            self.optional_info = '_'.join(optional_info)
        elif isinstance(optional_info, str):
            self.optional_info = optional_info
        elif isinstance(optional_info, dict):
            self.optional_info = '_'.join([str(key) + str(value) for key, value in optional_info.items()])
        else:
            raise ValueError('optional_info should be list, str or dict')

    @property
    def run_name(self):
        # 如果运行已经开始，返回当前运行的 run_name
        if self.running:
            return mlflow.active_run().info.run_name
        # 每次调用 run_name 时，都会根据当前属性构造 run_name
        components = [
            self.mission_name,
            self.model_name,
            self.dataset_name,
            self.optional_info,
            self.time
        ]
        # 过滤掉空字符串，确保组件不为空
        valid_components = [component for component in components if component]
        # 使用下划线连接所有非空组件
        return '_'.join(valid_components)

    @run_name.setter
    def run_name(self, value):
        # 如果需要，可以设置一个方法来允许外部修改 run_name 的一些组件
        # 这里的逻辑取决于您希望如何处理 run_name 的赋值
        pass
        # 例如，您可以将一个新的值分解并分别设置各个组件
        # components = value.split('_')
        # self.mission_name = components[0]
        # self.model_name = components[1]
        # ...等等

    def start_run(self):
        # 设置实验名称
        mlflow.set_experiment(self.exp_name)
        # 启动mlflow run
        print(text.text_in_box(f'Start mlflow run {self.run_name}'))
        self.local_run_id = mlflow.start_run(run_name=self.run_name)
        self.running = True

        # 记录运行参数
        self.log_params(vars(self.arg))

        # 初始化缓存文件夹
        if self.cache_dir:
            import os
            # 如果不为空则清空
            if os.path.exists(self.cache_dir):
                import shutil
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir)
                print(f'Clear cache dir: {self.cache_dir}')
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)



    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()

    @ensure_running
    def set_tag(self, tags):
        # 设置标签
        '''
        :param tags: dict
        '''
        for key in tags.keys():
            mlflow.set_tag(key, tags[key])

    def log_metric(self, stage, key, value, pre='', step=None):
        # 记录指标
        key = stage + '_' + pre + ('_' if pre != '' else '') + key
        mlflow.log_metric(key, value, step if step else self.epoch + 1)

    def log_metrics(self, stage, metrics=None, pre='', step=None, **kwargs):
        if not isinstance(metrics, dict):
            if metrics is not None:
                raise ValueError('metrics should be dict')
            metrics = kwargs
        # 记录指标
        new_metrics = {}
        for key in metrics.keys():
            new_key = stage + '_' + pre + ('_' if pre != '' else '') + key
            new_metrics[new_key] = float(metrics[key])

        mlflow.log_metrics(new_metrics, step if step else self.epoch + 1)

    def log_artifact_autott(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pathtt = self.cache_dir
            result_name = func(*args, path=pathtt, **kwargs)
            if result_name == None:
                return result_name
            # 返回 path+folders, 提取folder(去除path)并自动创建文件夹
            folder = result_name.replace(pathtt, '')
            mlflow.log_artifacts(result_name, folder)
            return result_name

        return wrapper  # 返回装饰器, 规范装饰器的使用

    def log_params(self, params, pre=''):
        # 记录参数

        mlflow.log_params(params)

    def save_model(self, model, epoch, path=None):
        # 保存模型
        if path is None:
            path = self.run_name + str(epoch + 1) + '.pth'
        mlflow.pytorch.save_model(model, path)
        mlflow.log_artifact(path)
