import logging
import queue
import threading

import pyaudio
import webrtcvad
from collections import deque

import ollama_client
from stt import FunasrSTT
from wav_handle import *
from config_reader import ConfigReader
from ollama_client import OllamaClient

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

text_speaker_id = queue.Queue()  # 当前的语音片缓存


def consecutive_check(check_func, check_times=2):
    consecutive_passes = 0

    def wrapper(*args, **kwargs):
        nonlocal consecutive_passes
        result = check_func(*args, **kwargs)
        if result:
            consecutive_passes += 1
            logging.info(f"检测合格，当前连续合格次数: {consecutive_passes}")
            if consecutive_passes == check_times:
                consecutive_passes = 0
                logging.info("达到指定连续合格次数，返回 True")
                return True
        else:
            prev_passes = consecutive_passes
            consecutive_passes = 0
            logging.info(f"检测不合格，之前连续合格次数: {prev_passes}，连续合格次数重置为 0")
        return False

    return wrapper


class AudioRecorder:
    def __init__(self):
        config_reader = ConfigReader('config.json')  # 假设配置文件名为 config.json，根据实际情况修改

        audio_settings = config_reader.get_audio_settings()
        vad_settings = config_reader.get_vad_settings()
        check_cycle_settings = config_reader.get_check_cycle_settings()
        speaker_id_settings = config_reader.get_speaker_id_settings()
        ollama_settings = config_reader.get_ollama_settings()
        funasr_settings = config_reader.get_funasr_settings()

        self.ollama = None
        self.ollama = ollama_client.OllamaClient(
                ollama_settings.get('url'),
                ollama_settings.get('model'),
                ollama_settings.get('prompt'),
                ollama_settings.get('temperature', 0.7),
                ollama_settings.get('top_p', 0.9)
        )


        self.p = None
        self.audio_stream = None
        self.audio_channels = audio_settings.get('audio_channels', 1)  # 单声道
        self.audio_rate = audio_settings.get('audio_rate', 16000)  # 音频采样率
        self.chunk = audio_settings.get('chunk', 1024)  # 音频块大小

        self.is_recording = None

        self.vad_mode = vad_settings.get('vad_mode', 3)  # VAD 模式 (0-3, 数字越大越敏感)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(self.vad_mode)

        self.audio_output_path = audio_settings.get('audio_output_path', './audio_output/')
        self.audio_frames = queue.Queue()  # 当前的语音片缓存
        self.audio_buffer_now = []
        self.segments_to_save = []  # 当前语音

        # ----- 以下为检测周期 -----
        # 音频缓冲区检查的时间周期，单位为秒
        self.check_buffer_cycle = check_cycle_settings.get('check_buffer_cycle', 0.1)
        # VAD 检测中语音活动的比例阈值（范围 0 - 1）
        self.check_vad_rate = 0.3
        # ----- 以下为惰性检测 -----
        # VAD 检测连续合格的次数阈值，增加此值可使开始检测到语音活动更具惰性
        self.check_vad_threshold = vad_settings.get('check_vad_threshold', 3)
        # 连续检测到无语音活动的次数阈值，增加此值可使判断语音停止更具惰性
        self.check_speak_alive_threshold = vad_settings.get('check_speak_alive_threshold', 3)

        self.stt = FunasrSTT(funasr_settings.get('thred_sv', 0.35))
        # 实时检测说话人相关参数，弃用
        # self.speaker_id_check_cycle = speaker_id_settings.get('speaker_id_check_cycle', 4)
        # self.speaker_id_check_threshold = speaker_id_settings.get('speaker_id_check_threshold', 2)

    def init_audio_stream(self):
        self.p = pyaudio.PyAudio()
        self.audio_stream = self.p.open(format=pyaudio.paInt16,
                                        channels=self.audio_channels,
                                        rate=self.audio_rate,
                                        input=True,
                                        frames_per_buffer=self.chunk)

    def stop_audio_stream(self):
        self.audio_stream.stop_audio_stream()
        self.audio_stream.close()
        self.p.terminate()

    def start_recording(self):
        self.is_recording = True
        self.init_audio_stream()
        thread = threading.Thread(target=self._record)
        thread.start()

    def _record(self):
        while self.is_recording:
            data = self.audio_stream.read(self.chunk)
            self.audio_frames.put(data)

    def loop_get_audio(self):
        # 使用 deque 来缓存音频数据，最大长度为 self.check_vad_threshold - 1 且不小于 1
        raw_audio_before = deque(maxlen=max(1, (self.check_vad_threshold - 1)))
        # raw_audio_others_speak = deque(maxlen=self.speaker_id_check_cycle * self.speaker_id_check_threshold)
        vad_check = consecutive_check(self.check_vad_activity, self.check_vad_threshold)
        no_vad_check = consecutive_check(lambda x: not self.check_vad_activity(x), self.check_speak_alive_threshold)
        # speaker_id_check = consecutive_check(lambda x: not self.stt.do_distinguish_person(x),
        #                                      self.speaker_id_check_threshold)
        vad_detected = False
        vad_activity_count = 0  # 记录连续有语音活动的次数
        # speaker_id_detection_started = False  # 标记是否开始说话人检测

        while True:
            self.audio_buffer_now.append(self.audio_frames.get())
            if not self.check_audio_length(self.audio_buffer_now):  # 判断长度是否达标
                continue

            raw_audio = b''.join(self.audio_buffer_now)
            raw_audio = resample_int16_audio_2(raw_audio)  # ！！！！注意此处将前端采集的44100采样率的音频转化成了16000！！！！
            self.audio_buffer_now = []

            if not vad_detected:
                vad_check_result = vad_check(raw_audio)
                if vad_check_result:  # 通过vad检测是否有语音活动
                    self.segments_to_save.append((raw_audio, time.time()))
                    vad_detected = True
                    print(f'检测到语音活动', vad_activity_count)
                else:  # 当前没有语音活动
                    # print(f'vad not pass')
                    # 向 deque 中添加元素，若超过最大长度会自动移除最旧元素
                    raw_audio_before.append((raw_audio, time.time()))
                    # raw_audio_others_speak.append((raw_audio, time.time()))
                    if not self.segments_to_save:  # 空数据，证明之前没有语音活动
                        pass
                    vad_activity_count = 0  # 无语音活动，重置计数
            else:
                no_vad_result = no_vad_check(raw_audio)
                if no_vad_result:
                    print('连续检测到无语音活动，认为语音停止')
                    # 计算需要保留的音频片段，这一段是为了将可能存在语音的头加回来
                    raw_audio_before_list = list(raw_audio_before)
                    print(len(raw_audio_before_list))
                    self.segments_to_save = raw_audio_before_list + self.segments_to_save
                    # 减去 (self.check_speak_alive_threshold - 2) 是为了在判断语音停止后多保留一些音频片段，确保数据的完整性
                    useful_length = len(self.segments_to_save) - (self.check_speak_alive_threshold - 2)
                    print(f'len{useful_length}, len{len(self.segments_to_save)}')

                    # self.segments_to_save = self.segments_to_save[:useful_length]
                    auido_segments_th = threading.Thread(target=self.handle_audio_segments)
                    auido_segments_th.start()
                    # self.handle_audio_segments()
                    vad_detected = False
                    # 清空 deque
                    raw_audio_before.clear()
                    vad_activity_count = 0  # 语音停止，重置计数
                    # speaker_id_detection_started = False  # 停止说话人检测
                else:
                    self.segments_to_save.append((raw_audio, time.time()))
                    vad_activity_count += 1
                    # print('count', vad_activity_count)
                    # -----以下为实时检测说话人，资源占用及其爆炸，请不要打开-----
                    # if vad_activity_count == self.speaker_id_check_cycle:
                    #     print('达到连续有语音活动次数，开始第一次说话人检测')
                    #     speaker_id_detection_started = True
                    #     vad_activity_count = 0
                    # if speaker_id_detection_started:
                    #     speaker_id_check_list = self.segments_to_save[-self.speaker_id_check_cycle:]
                    #     speaker_id_check_bytes = [seg[0] for seg in speaker_id_check_list]
                    #     speaker_id_check_wav_bytes = self.convert_to_wav_bytes(speaker_id_check_bytes)
                    #     speaker_id_check_result = speaker_id_check(speaker_id_check_wav_bytes)
                    #     if speaker_id_check_result:
                    #         print('连续检测达到指定次数，进行说话人切换')
                    #         raw_audio_others_speak_list = list(raw_audio_others_speak)
                    #         num_elements = len(raw_audio_others_speak_list) * 3 // 4
                    #         useful_length = len(self.segments_to_save) - num_elements
                    #         print('useful_length', useful_length)
                    #         self.segments_to_save = self.segments_to_save[:useful_length]
                    #         self.handle_audio_segments()
                    #         self.segments_to_save += raw_audio_others_speak_list[-useful_length:]
                    #         speaker_id_detection_started = False  # 完成说话人检测，重置标记
                    #         vad_activity_count = 0  # 重置语音活动计数

    def start_audio_handle_loop(self):
        threading.Thread(target=self.loop_get_audio).start()

    def handle_audio_segments(self):
        audio_frames = [seg[0] for seg in self.segments_to_save]
        self.segments_to_save = []
        audio_bytes = convert_to_wav_bytes(audio_frames)  # 进行 wav 转换
        audio_bytes_handle = self.stt.do_trans(audio_bytes)
        print('audio_bytes_handle', audio_bytes_handle)
        try:
            audio_bytes_chunk_list = audio_bytes_handle[0]["sentence_info"]
        except:
            print('垃圾信息')
            return
        print(audio_bytes_chunk_list)  # TODO 此处为stt输出的语音内容
        # speaker_id = self.stt.do_distinguish_person(audio_bytes)
        # print('speaker_id', speaker_id)

        audio_bytes_list = []
        text_list = []
        for chunk in audio_bytes_chunk_list:
            start = chunk['start']
            end = chunk['end']
            text = chunk['text']
            if ((end - start) < 200) or (len(text) <= 2):
                continue
            # 假设 audio_bytes 是字节数据，并且可以按时间戳切片
            sliced_audio = slice_wav_bytes(audio_bytes, start, end)
            audio_bytes_list.append(sliced_audio)
            text_list.append(text)
        # print(audio_bytes_list)
        # print('##', len(audio_bytes_list))
        # save_audio_file(audio_bytes, self.audio_output_path)

        speak_id_now = ''
        text_now = ''
        last_id = ''
        for i, one_audio_bytes in enumerate(audio_bytes_list):
            speak_id = self.stt.do_distinguish_person(one_audio_bytes)
            if speak_id_now == '':
                speak_id_now = speak_id
                text_now += text_list[i]
            elif speak_id_now != speak_id:
                speak_id_now = speak_id
                text_now = self.handle_text(text_now)
                text_speaker_id.put((speak_id_now, self.stt.voiceprint_dict[speak_id_now]['name'], text_now))
                text_now = ''
                text_now += text_list[i]
            else:
                text_now += text_list[i]
            # print('##', text_speaker_id)
            save_audio_file(self.stt.voiceprint_dict[speak_id]['wav_bytes'], self.audio_output_path, speak_id)
        if text_now != '':
            text_now = self.handle_text(text_now)
            text_speaker_id.put((speak_id_now, self.stt.voiceprint_dict[speak_id_now]['name'], text_now))
        return audio_frames

    def handle_text(self, text):
        trans = None
        try:
            trans = self.ollama.generate_response(text)
        except Exception as e:
            pass
        if trans:
            return text + trans
        else:
            return text


    def check_audio_length(self, audio_buffer):
        if len(audio_buffer) * self.chunk / self.audio_rate >= self.check_buffer_cycle:
            return True
        else:
            return False

    def check_vad_activity(self, audio_data):
        # 将音频数据分块检测
        num, rate = 0, self.check_vad_rate
        step = int(self.audio_rate * 0.02)  # 20ms 块大小
        flag_rate = round(rate * len(audio_data) // step)

        for i in range(0, len(audio_data), step):
            one_buf = audio_data[i:i + step]
            if len(one_buf) == step:
                if self.vad.is_speech(buf=one_buf,
                                      sample_rate=self.audio_rate):
                    num += 1
        if num > flag_rate:
            return True
        return False

    def check_speak_alive(self, dead_count):
        if dead_count >= self.check_speak_alive_threshold:
            return False
        else:
            return True

# a = AudioRecorder()
# a.start_recording()
# a.start_audio_handle_loop()
