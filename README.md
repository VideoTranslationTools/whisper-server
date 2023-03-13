# whisper-server
使用 openai whisper 进行语音转文本，然后再提交到翻译服务制作双语字幕

后续的机翻功能会用到，还需要配合 OpenAI 的 GPT（也许是某个其他的第三方服务） 做翻译服务。

## 安装支持

建议使用 conda 建立虚拟环境

```shell
# 日志库
pip install loguru
# whisper 非首次安装，升级
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
# whisperx 非首次安装，升级
pip install git+https://github.com/m-bain/whisperx.git --upgrade
# stable-ts 非首次安装，升级
pip install -U git+https://github.com/jianfch/stable-ts.git
# flask
pip install flask
```
