import time
from os import path
from typing import Iterator, TextIO

import torch
from loguru import logger

from pathlib import Path
import argparse
from flask import Flask, request, jsonify
from threading import Thread
from enum import Enum

import whisperx
from whisperx.utils import format_timestamp

model_size = "large-v2"

# ["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words,
# or merge them into neighbouring.")
INTERPOLATE_METHOD = "nearest"


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
parser.add_argument("--device", default='cuda', type=str)
# device_index gpu 的时候有效 "0,1,2,3" or "0"
parser.add_argument("--device_index", default="0", type=str)
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
# device index
# 将 "0,1,2,3" or "0" 转换为 [0,1,2,3] or [0]
if arg_dict['device_index'].find(",") != -1:
    g_device_index = [int(x) for x in arg_dict['device_index'].split(",")]
else:
    g_device_index = [int(arg_dict['device_index'])]


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
        # 如果设置了音频的语言，就使用设置的语言
        if jdata["language"] != "":
            tdata.language = jdata["language"]
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

        # 计算执行的耗时
        start_time = time.time()
        audio = whisperx.load_audio(tan_data.input_audio)
        batch_size = 16  # reduce if low on GPU mem
        # 开始转换，注意，下面这句话不会马上开始执行，而是在 segments 遍历的时候才真正开始
        transcribe_result = g_model.transcribe(audio=audio, batch_size=batch_size)
        logger.info("Transcription {task_id} aligning...", task_id=tan_data.task_id)
        # load alignment model and metadata
        # 优先使用外部指定的音频语言
        now_audio_language_code = tan_data.language
        if now_audio_language_code == "":
            # 如果外部没有指定，那么就用自动检测的语言
            now_audio_language_code = transcribe_result["language"]
        model_a, metadata = whisperx.load_align_model(language_code=now_audio_language_code,
                                                      device=arg_dict['device'])
        # align whisper output
        result_aligned = whisperx.align(transcribe_result["segments"], model_a, metadata, tan_data.input_audio,
                                        arg_dict['device'],
                                        interpolate_method=INTERPOLATE_METHOD,
                                        return_char_alignments=False,
                                        )
        logger.info("Transcription {task_id} aligned.", task_id=tan_data.task_id)

        # 构建输出的路径
        p = Path(tan_data.input_audio)
        out_srt_path = path.join(p.parent, p.stem + '.srt')

        with open(out_srt_path, "w", encoding="utf-8") as srt:
            whisperx_write_srt(result_aligned["segments"], file=srt)

        # 计算执行的耗时
        end_time = time.time()

        # 清理缓存
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Transcription:(transcribe) {task_id} end, time: {time} s", task_id=tan_data.task_id,
                    time=(end_time - start_time))

        logger.info("Transcription {task_id} saved to {file}.", task_id=tan_data.task_id, file=out_srt_path)

        # 任务状态设置为完成
        tan_data.task_status = TaskStatus.finished
        # 更新任务的状态
        g_task_dic[tan_data.task_id] = tan_data


# whisperx 写 srt 的方法
def whisperx_write_srt(transcript: Iterator[dict], file: TextIO, spk_colors=None):
    """
    Write a transcript to a file in SRT format.
    Example usage:
        from pathlib import Path
        from whisper.utils import write_srt
        result = transcribe(model, audio_path, temperature=temperature, **args)
        # save SRT
        audio_basename = Path(audio_path).stem
        with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
    """
    # spk_colors = {'SPEAKER_00':'white','SPEAKER_01':'yellow'}
    for i, segment in enumerate(transcript, start=1):
        # write srt lines

        text = f"{segment['text'].strip().replace('-->', '->')}"
        if spk_colors and 'speaker' in segment.keys():
            # f'<font color="{spk_colors[sentence.speaker]}">{text}</font>'
            text = f'<font color="{spk_colors[segment["speaker"]]}">{text}</font>'
        text += "\n"
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
            f"{text}",
            file=file,
            flush=True,
        )


if __name__ == '__main__':
    logger.info("Device: {device_name}", device_name=arg_dict['device'])
    # 加载模型
    logger.info("Loading model: {model_name} ...", model_name=arg_dict['model_size'])

    g_model = whisperx.load_model(arg_dict['model_size'], device=arg_dict['device'],
                                  compute_type=arg_dict['compute_type'])

    logger.info("Whisper model loaded.")

    # 启动任务线程
    logger.info("Start task thread...")
    t1 = Thread(target=task_transcribe)
    t1.start()

    logger.info("Try start server...")
    # 启动服务
    app.run(host='0.0.0.0', port=arg_dict['port'])
