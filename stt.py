import datetime
import uuid
import io

import torch
from funasr import AutoModel
import torchaudio
import os
from modelscope.pipelines import pipeline
from modelscope import snapshot_download
from wav_handle import *
from config_reader import SpeakerConfigReader

if torch.cuda.is_available():
    device = "cuda:0"  # 使用第一个 GPU
else:
    device = "cpu"
class FunasrSTT:
    def __init__(self, thred_sv=0.45):
        models = [
            'iic/speech_campplus_sv_zh-cn_16k-common',
            'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
            'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
            'iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
        ]
        for model in models:
            model_path = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "models", model)
            if not os.path.exists(model_path):
                snapshot_download(model)

        config_file_path = "speaker_config.json"
        config_reader = SpeakerConfigReader(config_file_path)

        # 获取读取到的配置信息
        speaker_config = config_reader.get_speaker_config()
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
                ngpu=self.config["ngpu"],
                ncpu=self.config["ncpu"],
                device=self.config["device"],
            )
        except Exception as e:
            print(f"模型初始化失败: {e}")

        self.sv_pipeline = pipeline(
            task='speaker-verification',
            model='D:/my_code/model/speech_campplus_sv_zh-cn_16k-common',
            model_revision='v1.0.0',
            device=device
        )

        self.result_queue = []

        self.voiceprint_dict = {}  # 每个说话人作为 key，value 为列表，列表中为当前说话人对应的每个音频片段
        self.voiceprint_dict.update(speaker_config)
        self.black_speaker = {}
        self.black_count = 1
        self.thred_sv = thred_sv

        self.count = 0

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

    def do_trans(self, audio_bytes):
        res = ''
        try:
            res = self.model.generate(input=audio_bytes,
                                      batch_size_s=100,
                                      is_final=True,
                                      sentence_timestamp=True
                                      )
        except Exception as e:
            print('error#:', e)
            return e
        return res

    def audio_bytes_to_numpy(self, audio_bytes):
        audio, _ = torchaudio.load(io.BytesIO(audio_bytes))
        audio = audio.squeeze().numpy()
        return audio

    def do_distinguish_person(self, audio_bytes):
        self.count += 1
        audio_np = self.audio_bytes_to_numpy(audio_bytes)
        for existing_user_id, existing_audio_dict in self.voiceprint_dict.items():
            existing_audio_np = self.audio_bytes_to_numpy(existing_audio_dict['wav_bytes'])
            result = self.sv_pipeline([audio_np, existing_audio_np], thr=self.thred_sv)
            sv_result = result['text']
            if sv_result == "yes":
                self.voiceprint_dict[existing_user_id]['wav_bytes'] = (
                    merge_wav_bytes(self.voiceprint_dict[existing_user_id]['wav_bytes'], audio_bytes))
                self.voiceprint_dict[existing_user_id]['voice_times'] += 1
                self.voiceprint_dict_lifecycle_management(existing_user_id)
                return existing_user_id

        user_id = str(uuid.uuid4())
        self.voiceprint_dict[user_id] = {
            'wav_bytes': audio_bytes,
            'name': '',
            'timestamp': datetime.datetime.now(),
            'voice_times': 1
        }
        self.voiceprint_dict_lifecycle_management(user_id)
        return user_id

    def voiceprint_dict_lifecycle_management(self, speaker_id_now):
        voiceprint_dict_new = {}
        del_speaker_id_list = []
        time_now = datetime.datetime.now()
        for user_id, one_voiceprint_dict in self.voiceprint_dict.items():
            if user_id == speaker_id_now:
                voiceprint_dict_new[speaker_id_now] = self.voiceprint_dict[user_id]
            else:
                if time_now - one_voiceprint_dict['timestamp'] > datetime.timedelta(minutes=10):
                    del_speaker_id_list.append(user_id)
        for speaker_id in del_speaker_id_list:
            del self.voiceprint_dict[speaker_id]
        sorted_dict_by_value = dict(sorted(self.voiceprint_dict.items(), key=lambda item: item[1]['voice_times'], reverse=True))
        voiceprint_dict_new.update(sorted_dict_by_value)
        self.voiceprint_dict = voiceprint_dict_new




