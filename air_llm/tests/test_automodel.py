import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

#sys.path.insert(0, '../airllm')

from ..airllm.auto_model import AutoModel



class TestAutoModel(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_auto_model_should_return_correct_model(self):
        mapping_dict = {
            'garage-bAInd/Platypus2-7B': ('LlamaForCausalLM', 'AirLLMBaseModel'),
            'Qwen/Qwen-7B': ('QWenLMHeadModel', 'AirLLMQWen'),
            'internlm/internlm-chat-7b': ('InternLMForCausalLM', 'AirLLMInternLM'),
            'THUDM/chatglm3-6b-base': ('ChatGLMForConditionalGeneration', 'AirLLMChatGLM'),
            'baichuan-inc/Baichuan2-7B-Base': ('BaichuanForCausalLM', 'AirLLMBaichuan'),
            'mistralai/Mistral-7B-Instruct-v0.1': ('MistralForCausalLM', 'AirLLMBaseModel'),
            'mistralai/Mixtral-8x7B-v0.1': ('MixtralForCausalLM', 'AirLLMBaseModel'),
        }

        architectures = {model_id: architecture for model_id, (architecture, _) in mapping_dict.items()}
        with patch(
            "air_llm.airllm.auto_model.AutoConfig.from_pretrained",
            side_effect=lambda model_id, **_kwargs: SimpleNamespace(
                architectures=[architectures[str(model_id)]]),
        ):
            for model_id, (_, expected_class) in mapping_dict.items():
                _module, actual_class = AutoModel.get_module_class(model_id)
                self.assertEqual(actual_class, expected_class, f"expecting {expected_class}")
