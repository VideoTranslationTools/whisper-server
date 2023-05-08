import time
from os import path

import pysubs2
from loguru import logger

from pathlib import Path
import argparse
from flask import Flask, request, jsonify
from threading import Thread
from enum import Enum

from faster_whisper import WhisperModel

model_size = "large-v2"


# 任务的状态
class TaskStatus(Enum):
    pending = 1
    running = 2
    finished = 3
    error = 4


# 参数解析
parser = argparse.ArgumentParser()
# device
'''
    cuda
    cpu
'''
parser.add_argument("--device", default='auto', type=str)
# compute_type
'''
choices=[
            "default",
            "int8",
            "int8_float16",
            "int16",
            "float16",
            "float32",
        ],
'''
parser.add_argument("--compute_type", default='default', type=str)
# GPU ID
parser.add_argument("--gpu_id", default='0', type=int)
# 模型
'''
MODEL_NAMES = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
]
'''
parser.add_argument("--model_size", default='large-v2', type=str)  # tiny base small medium large
# 启动的端口
parser.add_argument("--port", default='5000', type=int)
# http 接口的 token
parser.add_argument("--token", default='1234567890', type=str)
# 是否使用 VAD 进行语言的识别修正时间轴
parser.add_argument("--vad", default=True, type=bool)
# 解析参数
args = parser.parse_args()
# 将arg_dict转换为dict格式
arg_dict = args.__dict__

app = Flask(__name__)
# 全局模型的实例
g_model = ""
# 全局的任务字典
g_task_dic = {}
# 全局的任务列表
g_task_list = []
# http 接口的 token
g_token = arg_dict['token']


# 语音识别任务的数据结构
class TranscribeData:
    def __init__(self, task_id: int, input_audio_full_path: str):
        self.task_id = task_id
        self.input_audio = input_audio_full_path
        self.task_status = TaskStatus.pending
        self.language = ""


@app.route('/')
def hello():
    return 'hello world!'


@app.route('/transcribe', methods=["GET", "POST"])
def transcribe():
    if request.headers.get('Authorization'):
        get_token = request.headers['Authorization']
        if get_token == "":
            return jsonify({"code": 400, "msg": "token error"})
        tokens = str.split(get_token, " ")
        if len(tokens) != 2:
            return jsonify({"code": 400, "msg": "token error"})
        if tokens[1] != g_token:
            return jsonify({"code": 400, "msg": "token error"})
    else:
        return jsonify({"code": 400, "msg": "token error"})

    if request.method == 'GET':
        # 获取任务的 id
        task_id = request.args.get("task_id")
        # 获取任务的状态
        task_status = get_task_status(int(task_id))
        if task_status is None:
            return jsonify({"code": 400, "msg": "task not found"})
        else:
            return jsonify({"code": 200, "status": str(task_status.value)})

    elif request.method == 'POST':
        # 获取任务
        # 获取 JSON 数据
        jdata = request.get_json()
        # 判断文件是否存在
        if not Path(jdata["input_audio"]).exists():
            return jsonify({"code": 400, "msg": "file not found"})
        tdata = TranscribeData(jdata["task_id"], jdata["input_audio"])
        add_task(tdata)
        return jsonify({"code": 200, "msg": "ok"})
    else:
        return jsonify({"code": 400, "msg": "error"})


# 添加任务到队列中
def add_task(data: TranscribeData):
    global g_task_dic, g_task_list

    if data.task_id in g_task_dic:
        return
    # 任务不存在，添加到任务字典中
    g_task_dic[data.task_id] = data
    # 任务不存在，添加到任务列表中
    g_task_list.append(data)


# 获取任务的状态
def get_task_status(task_id: int):
    global g_task_dic

    if task_id not in g_task_dic:
        return None
    # 任务存在
    status = g_task_dic[task_id].task_status
    if status != TaskStatus.pending and status != TaskStatus.running:
        # 如果 任务状态不是 pending 和 running，说明任务已经完成，删除任务
        del g_task_dic[task_id]
    return status


# 任务线程
def task_transcribe():
    global g_task_dic, g_task_list
    # 循环获取任务
    while True:

        if len(g_task_list) == 0:
            # 休眠1秒
            time.sleep(1)
            continue
        # 从队列中取出一个任务执行
        tan_data = g_task_list.pop(0)
        # 任务状态设置为运行中
        tan_data.task_status = TaskStatus.running
        # 更新任务的状态
        g_task_dic[tan_data.task_id] = tan_data

        logger.info("Transcription {task_id} start...", task_id=tan_data.task_id)

        # 检测文件是否存在
        if not Path(tan_data.input_audio).exists():
            logger.error("File {file} not found.", file=tan_data.input_audio)
            tan_data.task_status = TaskStatus.error
            g_task_dic[tan_data.task_id] = tan_data
            continue

        # 开始转换，注意，下面这句话不会马上开始执行，而是在 segments 遍历的时候才真正开始
        segments, info = g_model.transcribe(audio=tan_data.input_audio, vad_filter=arg_dict['vad'])
        # to use pysubs2, the argument must be a segment list-of-dicts
        results = []
        index = 0
        for s in segments:
            print(str(index), s.start, s.end, s.text)
            segment_dict = {'start': s.start, 'end': s.end, 'text': s.text}
            results.append(segment_dict)
            index += 1

        subs = pysubs2.load_from_whisper(results)
        # 构建输出的路径
        p = Path(tan_data.input_audio)
        out_srt_path = path.join(p.parent, p.stem + '.srt')
        out_ass_path = path.join(p.parent, p.stem + '.ass')
        # save srt file
        subs.save(out_srt_path)
        # save ass file
        subs.save(out_ass_path)

        logger.info("Transcription {task_id} saved to {file}.", task_id=tan_data.task_id, file=out_srt_path)

        # 任务状态设置为完成
        tan_data.task_status = TaskStatus.finished
        # 更新任务的状态
        g_task_dic[tan_data.task_id] = tan_data


if __name__ == '__main__':

    logger.info("Device: {device_name}", device_name=arg_dict['device'])
    # 加载模型
    logger.info("Loading model: {model_name} ...", model_name=arg_dict['model_size'])

    g_model = WhisperModel(arg_dict['model_size'], device=arg_dict['device'], compute_type=arg_dict['compute_type'])

    logger.info("Whisper model loaded.")

    # 启动任务线程
    logger.info("Start task thread...")
    t1 = Thread(target=task_transcribe)
    t1.start()

    logger.info("Try start server...")
    # 启动服务
    app.run(host='0.0.0.0', port=arg_dict['port'])
