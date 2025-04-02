import json
from wav_handle import read_wav_to_bytes


class ConfigReader:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"错误: 未找到配置文件 {self.config_file_path}")
            return None
        except json.JSONDecodeError:
            print(f"错误: 配置文件 {self.config_file_path} 不是有效的 JSON 格式")
            return None

    def get_audio_settings(self):
        return self.config.get('audio_settings', {}) if self.config else {}

    def get_vad_settings(self):
        return self.config.get('vad_settings', {}) if self.config else {}

    def get_check_cycle_settings(self):
        return self.config.get('check_cycle_settings', {}) if self.config else {}

    def get_speaker_id_settings(self):
        return self.config.get('speaker_id_settings', {}) if self.config else {}

    def get_ollama_settings(self):
        return self.config.get('ollama_settings', {}) if self.config else {}

    def get_funasr_settings(self):
        return self.config.get('funasr_settings', {}) if self.config else {}


class SpeakerConfigReader:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.speaker_config = self._load_config()
        self._process_audio_files()

    def _load_config(self):
        try:
            with open(self.config_file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"错误: 未找到配置文件 {self.config_file_path}")
            return {}
        except json.JSONDecodeError:
            print(f"错误: 配置文件 {self.config_file_path} 不是有效的 JSON 格式")
            return {}

    def _process_audio_files(self):
        for speaker_id in self.speaker_config:
            audio_path = self.speaker_config[speaker_id].get('audio_path')
            if audio_path:
                wav_bytes = read_wav_to_bytes(audio_path)
                if wav_bytes:
                    self.speaker_config[speaker_id]['wav_bytes'] = wav_bytes
                    del self.speaker_config[speaker_id]['audio_path']

    def get_speaker_config(self):
        return self.speaker_config