import transformers
from modify_llama_vam import llama_model_forward_vam, llama_attn_forward_vam
from modify_mistral_vam import mistral_model_forward_vam, mistral_attn_forward_vam
from modify_qwen2_vam import qwen2_model_forward_vam, qwen2_attn_forward_vam



def replace_flashllama_attn_with_vamattn():
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward_vam
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_attn_forward_vam

def replace_flashmistral_attn_with_vamattn():
    transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward_vam
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_attn_forward_vam

def replace_flashqwen2_attn_with_vamattn():
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen2_model_forward_vam
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_attn_forward_vam

