# 语音处理与文本传输项目

## 项目简介

本项目是一个基于 WebSocket 的语音处理和文本传输系统，能够接收客户端的音频数据，进行语音识别和说话人区分，并将处理后的文本消息实时发送回客户端。

## 功能说明

1. **WebSocket 服务器**：`web.py` 文件启动一个 WebSocket 服务器，监听 `ws://0.0.0.0:8765` 地址，接收客户端发送的音频数据，并将处理后的文本消息发送回客户端。
2. **语音识别和说话人区分**：`stt.py` 文件中的 `FunasrSTT` 类实现了语音识别和说话人区分功能，能够根据音频数据识别文本并区分说话人。
3. **Ollama 交互**：`ollama_client.py` 文件中的 `OllamaClient` 类用于与 `Ollama` 模型进行交互，根据配置信息发送请求并获取响应。在原先的设计中主要用于进行文本翻译

## 安装依赖

```bash
pip install requests torchaudio websockets aioconsole funasr pyaudio webrtcvad modelscope[framework]
```

## 项目结构

```plaintext
WebSpeakerDiffProj/
├── config_reader.py      # 配置文件读取模块
├── ollama_client.py      # Ollama 模型交互模块
├── stt.py                # 语音识别和说话人区分模块
├── web.py                # WebSocket 服务器模块
├── config.json			  # 基础配置
├── speak_config.json     # 说话人配置
├── audio_output          # 语音输出
├── speak_audio           # 说话人配置参考语音建议存放文件夹
```

## 配置文件

在项目根目录下创建 `config.json` 和 `speaker_config.json` 文件，分别用于存储 `ollama` 相关配置和说话人配置信息。

### `config.json` 示例

```json
{
    "audio_settings": {
        "audio_channels": 1,
        "audio_rate": 16000,
        "chunk": 1024,
        "audio_output_path": "./audio_output/"
    },
    "vad_settings": {
        "vad_mode": 3,
        "check_vad_threshold": 3
    },
    "check_cycle_settings": {
        "check_buffer_cycle": 0.1,
        "speaker_id_check_cycle": 4
    },
    "speaker_id_settings": {
        "check_speak_alive_threshold": 3,
        "speaker_id_check_threshold": 2
    },
    "ollama_settings": {
        "url": "http://localhost:11434/api/generate",
        "model": "qwen2.5:0.5b",
        "prompt": "将此内容翻译成英语（不要带任何多余的话，不需要解释，只需要翻译即可）：",
        "temperature": 0.7,
        "top_p": 0.9
    },
    "funasr_settings": {
        "thred_sv": 0.35  # 重要参数，得分阈值来进行识别，阈值越高，判定为同一人的条件越严格
    }
}
```

### `speaker_config.json` 示例

```json
{
    "speaker1": {
        "audio_file": "path/to/audio/file1.wav"
    },
    "speaker2": {
        "audio_file": "path/to/audio/file2.wav"
    }
}
```

## 运行项目

```bash
python web.py
.\index.html  # 或者在任意局域网的其他设备中运行index.html。
```

## 注意事项

- 在局域网任意设备中运行html，但请在html文件中配置websocket地址，或者每次web运行时双击按钮在配置页面配置websocket地址
- 确保 `config.json` 和 `speaker_config.json` 文件中的配置信息正确。
- 运行项目前，确保所需的模型已经下载或可以正常访问，直接运行会尝试下载模型至默认地址。
- 若出现连接错误或其他异常，请检查网络连接和配置文件。

## 演示

![image-20250410193506807](/pic/image-20250410193506807.png)

