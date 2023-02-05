from transformer.src.transformers.models.mbart import MBartPreTrainedModel, MBartConfig, MBartModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
ENCODER_PROMPT_LEN = 100 ###lang: 10, intent:15
DECODER_PROMPT_LEN = 5

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens

class MBartForConditionalGeneration(MBartPreTrainedModel):
    
    def __init__(self, config: MBartConfig, encoder_prompt_len = ENCODER_PROMPT_LEN,args = None):
        super().__init__(config)
        self.model = MBartModel(config, encoder_prompt_len,args=args)
        self.args = args
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        decoder_len = None, ### the prefix len of decoder
        lang = None

    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        return_dict = True

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            lang = lang
            
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        shift_logits = lm_logits[..., :-1, :].contiguous()  ##[16, 99]
        shift_labels = decoder_input_ids[..., 1:].contiguous() ##[16, 99, 250027]
        loss_fct = CrossEntropyLoss()
        '''
        shift_logits_prefix = [shift_logits[:i:decoder_len[i]] for i in range(decoder_len.shape[0])]
        shift_labels_prefix =  [shift_labels[:i:decoder_len[i]] for i in range(decoder_len.shape[0])]
        
        #shift_logits_prefix = torch.stack(shift_logits_prefix)
        #print(shift_logits_prefix.shape)
        #exit()
        loss_prefix = [loss_fct(shift_logits_prefix[i].unsqueeze(0).view(-1,shift_logits_prefix[i].size(-1)),shift_labels_prefix[i].unsqueeze(0).view(-1)) for i in range(decoder_len.shape[0])]
        print(loss_prefix)
        exit()
        '''
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss,lm_logits

        
