import types
import torch

from typing import List, Optional, Tuple, Union

from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

logger = logging.get_logger(__name__)

def llama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    max_seqlens: Optional[torch.LongTensor] = None, 
    max_layer_num: Optional[int] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache):
        return_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                cu_seqlens=cu_seqlens, 
                max_seqlens=max_seqlens,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
            
        if max_layer_num is not None and idx == max_layer_num:
            break

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def llama_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None, 
    max_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        cu_seqlens=cu_seqlens,     
        max_seqlens=max_seqlens,  
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs

def llama_flash_attention_2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    max_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, 
        cu_seqlens = cu_seqlens, max_seqlens = max_seqlens, dropout=dropout_rate
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def _flash_attention_forward(
    self, query_states, key_states, value_states, attention_mask, query_length, 
    cu_seqlens = None, max_seqlens = None, dropout=0.0, softmax_scale=None
):

    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        causal = self.is_causal and query_length != 1

    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    elif cu_seqlens is not None:
        assert query_states.size(0) == 1
        assert key_states.size(0) == 1
        assert value_states.size(0) == 1
        
        attn_output_unpad = flash_attn_varlen_func(
            query_states.squeeze(0),
            key_states.squeeze(0),
            value_states.squeeze(0),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlens,
            max_seqlen_k=max_seqlens,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        attn_output = attn_output_unpad.unsqueeze(0)
    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
        )

    return attn_output

def model_patch(model):

    model.forward = types.MethodType(llama_model_forward, model)
    for l in model.layers:
        l.forward = types.MethodType(llama_decoder_layer_forward, l)
        l.self_attn.forward = types.MethodType(llama_flash_attention_2_forward, l.self_attn)
        l.self_attn._flash_attention_forward = types.MethodType(_flash_attention_forward, l.self_attn)

    return model

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/data2/MODELS/Meta-Llama-3-8B", use_fast = True)

    text = [
        """吴京本来给战狼3准备的对手是ISIS那样的YSL极端组织，毕竟那几年全世界都活在绿色恐怖下，理论上来说，殴打这帮人不会有什么问题。谁想得到没几年，“绿祸”思潮直接退下去了，世界变成了红绿联手抗蓝的局？现在吴京也是被尬在原地了，要不然改剧本直接去勇闯萝莉岛，要不然就只能把剧本封存假装这事儿没发生过。
        """,
        "Reward models are key in reinforcement learning from human feedback (RLHF) systems, aligning the model behavior with human preferences. Particularly in the math domain, there have been plenty of studies using reward models to align policies for improving reasoning capabilities. Recently, as the importance of reward models has been emphasized, RewardBench is proposed to understand their behavior. However, we figure out that the math subset of RewardBench has different representations between chosen and rejected completions, and relies on a single comparison, which may lead to unreliable results as it only see an isolated case. Therefore, it fails to accurately present the robustness of reward models, leading to a misunderstanding of its performance and potentially resulting in reward hacking. In this work, we introduce a new design for reliable evaluation of reward models, and to validate this, we construct RewardMATH",
        "はじめは椎名の表現者としての延命装置として生まれ、彼女が音楽と向き合って自身の音楽的成長を促すため設定したカリキュラムのようにスタートした[8][9]。そして伊澤一葉と浮雲が加入した際は、彼らのように「実力がある」と言われながらも普段はアンダーグラウンドやインディーズで活動している陽の目を見ない才能には、自分たちの内輪だけで循環するのではなくメジャーの場で勝負して欲しいと思い、自分の方から一緒にやってくれないかと誘った[10][11]。椎名の意識の変化に伴い、バンドも彼らを世間に通用させるまでの過程そのものをビジネスとすることを目的としたプロジェクトへと変化していった[10][11]。この過程について椎名は「学習機関に始まり（『教育』）、職業訓練校、研究室・実験室を経て（『大人 (アダルト)』〜『スポーツ』）、最終的にメンバー各々がそれぞれ独立して稼働できる生産工場となった（『大発見』）」と表現した[8][9][10][12]。",
        """RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg git-lfs libaio-dev
RUN git lfs install
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF
RUN python3 -m pip install --no-cache-dir -e ./transformers[dev,onnxruntime]

RUN cd transformers && python3 setup.py develop

ARG FRAMEWORK
ARG VERSION
        """
    ]
    input_ids_list = list()
    seperate_input_ids = list()
    for t in text:
        tt = tokenizer(t)["input_ids"]
        tt = [tokenizer.bos_token_id,] + tt + [tokenizer.eos_token_id]
        input_ids_list.extend(tt)
        seperate_input_ids.append(tt)

    input_ids = torch.tensor(input_ids_list)
    
    seq_pos = torch.where(input_ids == tokenizer.eos_token_id)[0] + 1
    cu_seqlens = torch.cat((torch.tensor([0]), seq_pos))
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    
    max_seq_lens = torch.max(seq_lens)
    
    print(cu_seqlens)

    model = AutoModel.from_pretrained(
        "/data2/MODELS/Meta-Llama-3-8B", 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to("cuda:0").eval()
    model = model_patch(model)
    
    cu_seqlens = cu_seqlens.to("cuda:0").to(torch.int32)
    input_ids = input_ids.to("cuda:0").unsqueeze(0)
    max_seq_lens = max_seq_lens.to("cuda:0").to(torch.int32)
    
    varlen_model_output = model(input_ids, cu_seqlens = cu_seqlens, max_seqlens = max_seq_lens)
    no_varlen_model_output = model(input_ids)
    
    seperate_input_ids = [torch.tensor(x).to("cuda:0").unsqueeze(0) for x in seperate_input_ids]
    seperate_model_output = list()
    for x in seperate_input_ids:
        seperate_model_output.append(model(x).last_hidden_state.squeeze(0))
        
    print(varlen_model_output.last_hidden_state.size())
    
    varlen_model_output_list = list()
    varlen_model_output = varlen_model_output.last_hidden_state.squeeze(0)
    cu_idx = cu_seqlens.tolist()
    for i in range(len(cu_idx) - 1):
        varlen_model_output_list.append(varlen_model_output[cu_idx[i]:cu_idx[i+1]])
        
    no_varlen_model_output_list = list()
    no_varlen_model_output = no_varlen_model_output.last_hidden_state.squeeze(0)
    cu_idx = cu_seqlens.tolist()
    for i in range(len(cu_idx) - 1):
        no_varlen_model_output_list.append(no_varlen_model_output[cu_idx[i]:cu_idx[i+1]])
        
    for i in range(len(text)):
        print(seperate_model_output[i].size())
        print(varlen_model_output_list[i].size())
        print(no_varlen_model_output_list[i].size())
        
        norm = torch.mean(torch.abs(seperate_model_output[i]))
        
        print(norm)
        print(torch.mean(torch.abs(seperate_model_output[i] - varlen_model_output_list[i])) / norm)
        print(torch.mean(torch.abs(seperate_model_output[i] - no_varlen_model_output_list[i])) / norm)
        
        print(seperate_model_output[i] - varlen_model_output_list[i])
        print(seperate_model_output[i] - no_varlen_model_output_list[i])
        
        print()