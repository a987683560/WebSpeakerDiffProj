import io
import os
from scipy import signal
import struct
import time
import wave

import numpy as np
from pydub import AudioSegment
import librosa


def merge_wav_bytes(wav_bytes_1, wav_bytes_2):
    # 读取第一个 WAV 文件头
    chunk_id_1 = wav_bytes_1[0:4]
    chunk_size_1 = struct.unpack('<I', wav_bytes_1[4:8])[0]
    format_1 = wav_bytes_1[8:12]
    subchunk1_id_1 = wav_bytes_1[12:16]
    subchunk1_size_1 = struct.unpack('<I', wav_bytes_1[16:20])[0]
    audio_format_1 = struct.unpack('<H', wav_bytes_1[20:22])[0]
    num_channels_1 = struct.unpack('<H', wav_bytes_1[22:24])[0]
    sample_rate_1 = struct.unpack('<I', wav_bytes_1[24:28])[0]
    byte_rate_1 = struct.unpack('<I', wav_bytes_1[28:32])[0]
    block_align_1 = struct.unpack('<H', wav_bytes_1[32:34])[0]
    bits_per_sample_1 = struct.unpack('<H', wav_bytes_1[34:36])[0]
    subchunk2_id_1 = wav_bytes_1[36:40]
    subchunk2_size_1 = struct.unpack('<I', wav_bytes_1[40:44])[0]

    # 读取第二个 WAV 文件头
    chunk_id_2 = wav_bytes_2[0:4]
    chunk_size_2 = struct.unpack('<I', wav_bytes_2[4:8])[0]
    format_2 = wav_bytes_2[8:12]
    subchunk1_id_2 = wav_bytes_2[12:16]
    subchunk1_size_2 = struct.unpack('<I', wav_bytes_2[16:20])[0]
    audio_format_2 = struct.unpack('<H', wav_bytes_2[20:22])[0]
    num_channels_2 = struct.unpack('<H', wav_bytes_2[22:24])[0]
    sample_rate_2 = struct.unpack('<I', wav_bytes_2[24:28])[0]
    byte_rate_2 = struct.unpack('<I', wav_bytes_2[28:32])[0]
    block_align_2 = struct.unpack('<H', wav_bytes_2[32:34])[0]
    bits_per_sample_2 = struct.unpack('<H', wav_bytes_2[34:36])[0]
    subchunk2_id_2 = wav_bytes_2[36:40]
    subchunk2_size_2 = struct.unpack('<I', wav_bytes_2[40:44])[0]

    # 检查两个 WAV 文件的参数是否一致
    if (
            chunk_id_1 != chunk_id_2 or
            format_1 != format_2 or
            subchunk1_id_1 != subchunk1_id_2 or
            audio_format_1 != audio_format_2 or
            num_channels_1 != num_channels_2 or
            sample_rate_1 != sample_rate_2 or
            byte_rate_1 != byte_rate_2 or
            block_align_1 != block_align_2 or
            bits_per_sample_1 != bits_per_sample_2 or
            subchunk2_id_1 != subchunk2_id_2
    ):
        raise ValueError("两个 WAV 文件的参数不一致，无法合并。")

    # 计算合并后的数据块大小和文件总大小
    new_subchunk2_size = subchunk2_size_1 + subchunk2_size_2
    new_chunk_size = chunk_size_1 + subchunk2_size_2

    # 构建新的 WAV 文件头
    new_wav_header = chunk_id_1 + struct.pack('<I', new_chunk_size) + format_1 + subchunk1_id_1 + struct.pack(
        '<I', subchunk1_size_1) + struct.pack('<H', audio_format_1) + struct.pack('<H', num_channels_1) + struct.pack(
        '<I', sample_rate_1) + struct.pack('<I', byte_rate_1) + struct.pack('<H', block_align_1) + struct.pack(
        '<H', bits_per_sample_1) + subchunk2_id_1 + struct.pack('<I', new_subchunk2_size)

    # 提取两个 WAV 文件的数据部分
    data_start_1 = 44
    data_start_2 = 44
    data_1 = wav_bytes_1[data_start_1:]
    data_2 = wav_bytes_2[data_start_2:]

    # 合并数据
    merged_data = data_1 + data_2

    # 组合新的 WAV 文件
    new_wav_bytes = new_wav_header + merged_data

    return new_wav_bytes


def slice_wav_bytes(wav_bytes, start_time, end_time):
    # 读取 WAV 文件头
    chunk_id = wav_bytes[0:4]
    chunk_size = struct.unpack('<I', wav_bytes[4:8])[0]
    format = wav_bytes[8:12]
    subchunk1_id = wav_bytes[12:16]
    subchunk1_size = struct.unpack('<I', wav_bytes[16:20])[0]
    audio_format = struct.unpack('<H', wav_bytes[20:22])[0]
    num_channels = struct.unpack('<H', wav_bytes[22:24])[0]
    sample_rate = struct.unpack('<I', wav_bytes[24:28])[0]
    byte_rate = struct.unpack('<I', wav_bytes[28:32])[0]
    block_align = struct.unpack('<H', wav_bytes[32:34])[0]
    bits_per_sample = struct.unpack('<H', wav_bytes[34:36])[0]

    # 找到数据块的起始位置
    subchunk2_id = wav_bytes[36:40]
    subchunk2_size = struct.unpack('<I', wav_bytes[40:44])[0]

    # 将毫秒转换为秒
    start_time = start_time / 1000
    end_time = end_time / 1000

    # 计算起始和结束的字节偏移量
    start_byte = int(start_time * byte_rate)
    end_byte = int(end_time * byte_rate)

    # 确保偏移量在数据块范围内
    if start_byte > subchunk2_size:
        start_byte = subchunk2_size
    if end_byte > subchunk2_size:
        end_byte = subchunk2_size

    # 计算新的数据块大小
    new_subchunk2_size = end_byte - start_byte
    new_chunk_size = chunk_size - (subchunk2_size - new_subchunk2_size)

    # 构建新的 WAV 文件头
    new_wav_header = chunk_id + struct.pack('<I', new_chunk_size) + format + subchunk1_id + struct.pack(
        '<I', subchunk1_size) + struct.pack('<H', audio_format) + struct.pack('<H', num_channels) + struct.pack(
        '<I', sample_rate) + struct.pack('<I', byte_rate) + struct.pack('<H', block_align) + struct.pack(
        '<H', bits_per_sample) + subchunk2_id + struct.pack('<I', new_subchunk2_size)

    # 提取切割后的数据
    data_start = 44
    sliced_data = wav_bytes[data_start + start_byte:data_start + end_byte]

    # 组合新的 WAV 文件
    new_wav_bytes = new_wav_header + sliced_data

    return new_wav_bytes


def save_audio_file(audio_bytes, audio_output_path, speaker_id='0', audio_channels=1, audio_rate=16000):
    """
    将音频数据保存为 WAV 文件到指定目录
    """
    file_name = os.path.join(audio_output_path, f"{speaker_id}_{int(time.time())}.wav")
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(audio_channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(audio_rate)
        wf.writeframes(audio_bytes)
    print(f"音频文件已保存到: {file_name}")


def convert_to_wav_bytes(audio_frames, audio_channels=1, audio_rate=16000):
    virtual_file = io.BytesIO()
    wf = wave.open(virtual_file, 'wb')
    wf.setnchannels(audio_channels)
    wf.setsampwidth(2)  # 16-bit PCM
    wf.setframerate(audio_rate)
    wf.writeframes(b''.join(audio_frames))
    wf.close()
    virtual_file.seek(0)
    return virtual_file.read()


def resample_int16_audio(audio_data, input_rate=44100, output_rate=16000):
    # 如果 audio_data 是 bytes 类型，将其转换为 numpy 数组
    if isinstance(audio_data, bytes):
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
    else:
        audio_array = audio_data

    # 将 int16 数据转换为 float32 以进行重采样
    audio_float = audio_array.astype(np.float32) / 32767.0
    # 计算重采样后的长度
    num_samples = int(len(audio_float) * output_rate / input_rate)
    # 使用 scipy.signal.resample 进行重采样
    resampled_float = signal.resample(audio_float, num_samples)
    # 将重采样后的 float32 数据转换回 int16
    resampled_int16 = (resampled_float * 32767).astype(np.int16)
    return resampled_int16


def resample_int16_audio_2(audio_data, input_rate=44100, output_rate=16000):
    # 如果 audio_data 是 bytes 类型，将其转换为 numpy 数组
    if isinstance(audio_data, bytes):
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
    else:
        audio_array = audio_data

    # 将 int16 数据转换为 float32 以进行重采样
    audio_float = audio_array.astype(np.float32) / 32767.0

    # 使用 librosa 进行重采样
    resampled_float = librosa.resample(y=audio_float, orig_sr=input_rate, target_sr=output_rate)

    # 将重采样后的 float32 数据转换回 int16
    resampled_int16 = (resampled_float * 32767).astype(np.int16)
    return resampled_int16


def read_wav_to_bytes(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            wav_bytes = wf.readframes(wf.getnframes())
            return wav_bytes
    except FileNotFoundError:
        print(f"错误: 未找到文件 {file_path}")
        return None
    except wave.Error as e:
        print(f"错误: 读取 WAV 文件时出现问题: {e}")
        return None
