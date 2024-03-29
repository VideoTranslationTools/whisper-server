# whisper-server

> 以下只针对 Windows 版本的部署

使用 openai whisper 进行语音转文本，然后再提交到翻译服务制作双语字幕

后续的机翻功能会用到，还需要配合 OpenAI 的 GPT（也许是某个其他的第三方服务） 做翻译服务。

## 安装支持

首先在物理机安装：

* CUDA 11.7
* CUDNN 8

然后记得把 CUDNN 的 bin 目录加入到环境变量。然后重启。

然后安装 ffmpeg

建议使用 conda 建立虚拟环境

```shell
# 安装 torch GPU 版本
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 安装 whisperx 核心
pip install git+https://github.com/m-bain/whisperx.git@v3

# 或者是更新 whisperx 核心
pip install git+https://github.com/m-bain/whisperx.git@v3 --upgrade

# 日志库
pip install loguru
# flask
pip install flask
```

## 可能遇到的问题

Q：解决cudnn_cnn_infer64_8.dll 不在path中

A：去这个地方下载 https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-windows  得到 zlib123dllx64.zip 文件。然后解压得到 zlibwapi.dll，放到系统的 PATH 目录中（我个人放到了 CUDNN 目录中）

Q：安装完支持库后，运行提示 CUDA 无法启用什么的

A：再重装一次 torch···pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
