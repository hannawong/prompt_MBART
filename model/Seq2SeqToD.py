
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from model.GPT2Adapter import GPT2Adapter
from collections import defaultdict

class Seq2SeqToD(nn.Module):

    def __init__(self,args,adapter_num = 40):
        super().__init__()
        model = GPT2Adapter.from_pretrained("gpt2")
        model.args = args

        self.model = model

    def compute_PPL(self,input_ids,label,task_id=-1,device='cuda'):
        with torch.no_grad():
            lm_logits, *_ = self.model(
                            input_ids=input_ids.to(device),
                            attention_mask=None,
                            labels=None,
                            task_id=task_id
                            )
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = label.to(device)[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = torch.reshape(loss, shift_labels.size())
        return (loss.sum(1)/(loss!=0).sum(1)).tolist()

    def forward(self, input_ids, labels = None, task_id = -1,attention_mask = None,position_ids=None,past_key_values = None,s = -1,with_adapter = True,last_hidden = False, input_ids_prev = None,labels_prev = None,mix_layer = None,BNM = False,length = None,is_replay = None):
        loss = self.model(input_ids=input_ids,labels=labels,task_id=task_id,attention_mask = attention_mask,position_ids = position_ids,past_key_values = past_key_values,s = s,with_adapter = with_adapter,last_hidden = last_hidden,input_ids_prev = input_ids_prev,labels_prev = labels_prev,mix_layer = mix_layer, BNM = BNM,length = length,is_replay = is_replay)
        return loss
