# whisper-server
使用 openai whisper 进行语音转文本，然后再提交到翻译服务制作双语字幕

后续的机翻功能会用到，还需要配合 OpenAI 的 GPT（也许是某个其他的第三方服务） 做翻译服务。

基于以下几个库（稳定后会给出创建虚拟环境的脚本和具体库版本）：

* loguru
* whisper
* stable_whisper
* flask
