from transformer.src.transformers.models.mbart import MBartPreTrainedModel, MBartConfig, MBartModel
from transformers import AutoModelForMaskedLM
from transformer.src.transformers.models.bert import BertModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
ENCODER_PROMPT_LEN = 10 ###lang: 10, intent:15
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


class MBartDictionGeneration(nn.Module):
    def __init__(self,config,args) -> None:
        super().__init__()
        self.diction_bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25",config = config, args = args)
        self.config = config
        self.args = args
        self.dic_dim = 10
        self.attn_head_num = 10
        self.diction_attn_linear_layer = nn.Linear(768, 256)
        self.diction_attn_linear_layer1 = nn.Linear(256, self.attn_head_num * self.dic_dim)
        self.dictionary = nn.Parameter(torch.randn(self.dic_dim,400))
        self.encoder_layers = config.encoder_layers
        self.sigmoid = nn.Sigmoid()


    def forward(self,
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
        lang = None,
        input_ids_bert = None):


        dictionary_normed = torch.nn.functional.normalize(self.dictionary,p=2,dim=1)
        bert_cls_last_hidden_state = self.diction_bert_model(input_ids_bert)[0][:,0,:] ##[bz,768]
        bert_attention_score = self.diction_attn_linear_layer(bert_cls_last_hidden_state)##[bz, attn_head_num * dic_dim]
        bert_attention_score = torch.nn.functional.relu(bert_attention_score)
        bert_attention_score = self.diction_attn_linear_layer1(bert_attention_score)

        bert_attention_score = bert_attention_score.reshape((bert_attention_score.shape[0],self.attn_head_num,self.dic_dim))
        softmax = nn.Softmax(dim = 2)
        bert_attention_score = softmax(bert_attention_score) ##[bz,attn_head_num, dic_dim]
        prompt = torch.bmm(bert_attention_score,dictionary_normed.repeat(bert_attention_score.shape[0],1,1)) ##[bz,attn_head_num,800*layer_num]
        prompt = prompt.reshape((prompt.shape[0],self.attn_head_num,400)) ##[bz,attn_head_num,layer_num,800]
        ########### prompt generated!! #############
        loss,lm_logits = self.bart_model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask,decoder_len = decoder_len, lang = lang, prompt = prompt)
        orth_loss = torch.norm(torch.matmul(dictionary_normed.T,dictionary_normed))
        theta0 = 0.5
        theta1 = -1
        sparse_loss = torch.norm(torch.sigmoid(theta0*bert_attention_score+theta1),1)

        loss = loss + 0.005 * orth_loss + 0.0005*sparse_loss
        
        
        
        return loss,lm_logits

        



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
        lang = None,
        prompt = None

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
            lang = lang,
            prompt = prompt
            
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        shift_logits = lm_logits[..., :-1, :].contiguous()  ##[16, 99]
        shift_labels = decoder_input_ids[..., 1:].contiguous() ##[16, 99, 250027]
        loss_fct = CrossEntropyLoss()

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss,lm_logits

        
