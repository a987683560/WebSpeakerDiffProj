#-----for funasr-----
import os
import uuid

from funasr import AutoModel
import ffmpeg
#-----for audio recoder-----
import shutil

import cv2
import pyaudio
import wave
import threading
import numpy as np
import time
from queue import Queue

import torchaudio
import webrtcvad
import os
import threading
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
from funasr import AutoModel
import pygame
import edge_tts
import asyncio
from time import sleep
import langid
from langdetect import detect
import re
import requests
from pypinyin import pinyin, Style
from modelscope.pipelines import pipeline


class AudioRecorder:
    def __init__(self):
        self.AUDIO_RATE = 16000  # 音频采样率
        self.AUDIO_CHANNELS = 1  # 单声道
        self.CHUNK = 1024  # 音频块大小
        self.NO_SPEECH_THRESHOLD = 1

        self.VAD_MODE = 0  # VAD 模式 (0-3, 数字越大越敏感)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(self.VAD_MODE)

        self.p = None
        self.audio_stream = None
        self.audio_buffer = []
        self.segments_to_save = []
        self.saved_intervals = []
        self.last_active_time = time.time()
        self.last_vad_end_time = 0

        self.audio_output_path = './audio_output/'

        self.STT = FunasrSTT()

    def init_audio_stream(self):
        self.p = pyaudio.PyAudio()
        self.audio_stream = self.p.open(format=pyaudio.paInt16,
                                        channels=self.AUDIO_CHANNELS,
                                        rate=self.AUDIO_RATE,
                                        input=True,
                                        frames_per_buffer=self.CHUNK)

    def do_stop_stream(self):
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.p.terminate()

    def loop_get_audio(self):
        while 1:
            data = self.audio_stream.read(self.CHUNK)
            self.audio_buffer.append(data)

            # 每 0.5 秒检测一次 VAD
            if len(self.audio_buffer) * self.CHUNK / self.AUDIO_RATE >= 0.5:
                # 拼接音频数据并检测 VAD
                raw_audio = b''.join(self.audio_buffer)
                vad_result = self.handle_check_vad_activity(raw_audio)

                if vad_result:
                    print("检测到语音活动")
                    self.last_active_time = time.time()
                    self.segments_to_save.append((raw_audio, time.time()))
                else:
                    # print("静音中...")
                    pass
                self.audio_buffer = []  # 清空缓冲区

            # 检查无效语音时间
            if time.time() - self.last_active_time > self.NO_SPEECH_THRESHOLD:
                # 检查是否需要保存
                if self.segments_to_save and self.segments_to_save[-1][1] > self.last_vad_end_time:
                    self.do_save_audio()  #可能需要发送一个消息出去。
                    self.last_active_time = time.time()
                else:
                    pass

    def do_save_audio(self):
        audio_frames = [seg[0] for seg in self.segments_to_save]
        timestamp = str(time.time())
        if not os.path.exists(self.audio_output_path):
            os.makedirs(self.audio_output_path)
        save_path = self.audio_output_path + timestamp + '.wav'
        wf = wave.open(self.audio_output_path + timestamp + '.wav', 'wb')
        wf.setnchannels(self.AUDIO_CHANNELS)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(self.AUDIO_RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()
        self.segments_to_save.clear()
        print(f"音频保存至 {save_path}")
        self.STT.do_distinguish_person(save_path)

    def handle_check_replicate(self):
        start_time = self.segments_to_save[0][1]
        end_time = self.segments_to_save[-1][1]
        self.saved_intervals.append((start_time, end_time))
        # 检查是否与之前的片段重叠
        if self.saved_intervals and self.saved_intervals[-1][1] >= start_time:
            print("当前片段与之前片段重叠，跳过保存")
            self.segments_to_save.clear()
            return

    def handle_check_vad_activity(self, audio_data):
        # 将音频数据分块检测
        num, rate = 0, 0.5
        step = int(self.AUDIO_RATE * 0.02)  # 20ms 块大小
        flag_rate = round(rate * len(audio_data) // step)

        for i in range(0, len(audio_data), step):
            chunk = audio_data[i:i + step]
            if len(chunk) == step:
                if self.vad.is_speech(chunk, sample_rate=self.AUDIO_RATE):
                    num += 1

        if num > flag_rate:
            return True
        return False


class FunasrSTT:
    def __init__(self):
        # 配置信息
        self.config = {
            "home_directory": os.path.expanduser("~"),
            "model_revision": "v2.0.4",
            "ngpu": 1,
            "device": "cuda",
            "ncpu": 4
        }

        # 模型名称和路径配置
        self.model_names = {
            "asr": "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "vad": "speech_fsmn_vad_zh-cn-16k-common-pytorch",
            "punc": "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            "spk": "speech_campplus_sv_zh-cn_16k-common"
        }

        # 生成模型路径
        self.model_paths = {
            key: self._get_model_path(name) for key, name in self.model_names.items()
        }

        try:
            # 初始化模型
            self.model = AutoModel(
                model=self.model_paths["asr"],
                model_revision=self.config["model_revision"],
                vad_model=self.model_paths["vad"],
                vad_model_revision=self.config["model_revision"],
                punc_model=self.model_paths["punc"],
                punc_model_revision=self.config["model_revision"],
                spk_model=self.model_paths["spk"],
                spk_model_revision=self.config["model_revision"],
                # vad_kwargs={"max_single_segment_time": 500},
                ngpu=self.config["ngpu"],
                ncpu=self.config["ncpu"],
                device=self.config["device"],
                disable_pbar=True,
                disable_log=True,
                disable_update=True
            )
        except Exception as e:
            print(f"模型初始化失败: {e}")

        self.sv_pipeline = pipeline(
            task='speaker-verification',
            model='D:\my_code\model\speech_campplus_sv_zh-cn_16k-common',
            model_revision='v1.0.0'
        )

        self.result_queue = []
        self.speaker_audios = {}  # 每个说话人作为 key，value 为列表，列表中为当前说话人对应的每个音频片段

        self.voiceprint_dict = {}

    def _get_model_path(self, model_name):
        """
        生成模型的完整路径
        """
        return os.path.join(
            self.config["home_directory"],
            ".cache",
            "modelscope",
            "hub",
            "models",
            "iic",
            model_name
        )

    def do_trans(self, audio):
        asr_result_text = ''
        if os.path.exists(audio):
            audio_name = os.path.splitext(os.path.basename(audio))[0]
            _, audio_extension = os.path.splitext(audio)
            # 音频预处理
            try:
                audio_bytes, _ = (
                    ffmpeg.input(audio, threads=0, hwaccel='cuda')
                    .output("-", format="wav", acodec="pcm_s16le", ac=1, ar=16000)
                    .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
                )
                # print(f'audio{audio_bytes}')
                res = self.model.generate(input=audio_bytes,
                                          batch_size_s=100,
                                          is_final=True,
                                          sentence_timestamp=True,
                                          # thr=0.8
                                          )
                print(f'res{res}')
                rec_result = res[0]
                asr_result_text = rec_result['text']
                # print(f'rec_result: {rec_result}')
                print(f'asr_result_text: {asr_result_text}')
            except Exception as e:
                print(e)
        return asr_result_text

    def do_distinguish_person(self, audio):
        trans_result = self.do_trans(audio)
        if not trans_result:
            return ''
        for existing_audio in self.voiceprint_dict.keys():
            result = self.sv_pipeline([audio, existing_audio])
            sv_result = result['text']
            if sv_result == "yes":
                print('###', self.voiceprint_dict)
                return self.voiceprint_dict[existing_audio]

        user_id = str(uuid.uuid4())
        self.voiceprint_dict[audio] = user_id
        print('###', self.voiceprint_dict)
        return user_id


audio_recorder = AudioRecorder()
audio_recorder.init_audio_stream()
audio_recorder.loop_get_audio()
# audio_thread = threading.Thread(target=)
