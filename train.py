from __future__ import absolute_import, division, print_function
import argparse
import os
import random,json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm, trange
from model.GPT2Adapter import MBartForConditionalGeneration, MBartDictionGeneration
from transformer.src.transformers.models.mbart import MBartConfig
from transformers import (AdamW,AutoTokenizer)
                                
MODEL_CLASSES_DICT = {
    'mbart': (MBartConfig, MBartDictionGeneration,AutoTokenizer)
}
MODEL_CLASSES = {
    'mbart':(MBartConfig,MBartForConditionalGeneration,AutoTokenizer)
}


IN_MAX_SEQ = 50
OUT_MAX_SEQ = 100


from model.GPT2Adapter import ENCODER_PROMPT_LEN, DECODER_PROMPT_LEN


class TextSeqDataset(Dataset):  
    def __init__(self, args, tokenizer, in_max_seq=IN_MAX_SEQ, out_max_seq = OUT_MAX_SEQ, mode="train",langs = None):
        self.input_ids = []
        self.input_ids_bert = []
        self.decoder_input_ids = []
        self.attention_mask = []
        self.decoder_attention_mask = []
        self.decoder_len = []
        self.input_len = []
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.lang = [] ## since the language prediction acc is 1.0, we can use the oracle language id
        self.lang_dic = {"en":0,"fr":1,"de":2,"es":3,"zh":4,"hi":5,"ja":6,"pt":7,"tr":8}
        
        if mode == "train":
            if args.train_data_file == "all": self.langs = list(self.lang_dic.keys())
            else: self.langs = args.train_data_file.split(",")
        elif mode == "test":
            if args.eval_data_file == "all": self.langs = list(self.lang_dic.keys())
            else: self.langs = args.eval_data_file.split(",")
        
        if langs: self.langs = langs

        if args.task == "intent": task = "processed_intent"
        elif args.task == "joint": task = "processed_joint"
        elif args.task == "slot": task = "processed_slots"

        encoder_prompt_input_ids = [i for i in range(ENCODER_PROMPT_LEN)]
        for lang in self.langs:
            with open("./MultiATIS_data/"+task+"/"+mode+"_"+lang.upper()+".input", encoding="utf-8") as f:
                for line in tqdm(f):
                    line = line.strip()
                    raw_str = line.lower() ##inform ( name = hakka restaurant ; pricerange = moderate ) & hakka restaurant is moderate -ly priced
                    if len(raw_str.split()) > in_max_seq -1: 
                        raw_str = ' '.join(raw_str.split()[:in_max_seq -1])
                    raw_str += ' ' + tokenizer.eos_token 
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_str))
                    tokenized_text_bert = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(bert_tokenizer.cls_token+raw_str))
                    self.input_len.append(min(len(tokenized_text),in_max_seq))
                    attention_mask = [1] *  in_max_seq
                    if len(tokenized_text) < in_max_seq: 
                        attention_mask[-(in_max_seq - len(tokenized_text)):] = [0] * (in_max_seq - len(tokenized_text))
                        tokenized_text = tokenized_text + [0] * (in_max_seq - len(tokenized_text))  ###补零
                    else:
                        tokenized_text = tokenized_text[:in_max_seq]
                        
                    if len(tokenized_text_bert) < in_max_seq:
                        tokenized_text_bert = tokenized_text_bert + [0] * (in_max_seq - len(tokenized_text_bert))
                    else:
                        tokenized_text_bert = tokenized_text_bert[:in_max_seq]

                    tokenized_text = encoder_prompt_input_ids + tokenized_text
                    attention_mask = [1] * ENCODER_PROMPT_LEN + attention_mask
                    self.input_ids.append(tokenized_text)
                    self.attention_mask.append(attention_mask)
                    self.input_ids_bert.append(tokenized_text_bert)
                
        for lang in self.langs:
            with open("./MultiATIS_data/"+task+"/"+mode+"_"+lang.upper()+".output", encoding="utf-8") as f:
                decoder_prompt_input_ids = [i for i in range(DECODER_PROMPT_LEN)]
                for line in tqdm(f):
                    line = line.strip()
                    raw_str = line.lower() 
                    if len(raw_str.split()) > out_max_seq -1: 
                        raw_str = ' '.join(raw_str.split()[:out_max_seq -1])
                    raw_str_real = raw_str.split("@")[0].strip()
                    self.decoder_len.append(min(len(tokenizer.tokenize(raw_str_real)),out_max_seq))
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_str))
                    
            
                    attention_mask = [1] *  out_max_seq
                    if len(tokenized_text) < out_max_seq: 
                        attention_mask[-(out_max_seq - len(tokenized_text)):] = [0] * (out_max_seq - len(tokenized_text))
                        tokenized_text = tokenized_text + [0] * (out_max_seq - len(tokenized_text))  ###补零
                    else:
                        tokenized_text = tokenized_text[:out_max_seq]
                    self.decoder_input_ids.append(tokenized_text)
                    self.decoder_attention_mask.append(attention_mask)
                    self.lang.append(self.lang_dic[lang])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return torch.tensor(self.input_ids[item]), torch.tensor(self.attention_mask[item]), torch.tensor(self.decoder_input_ids[item]), torch.tensor(self.decoder_attention_mask[item]), torch.tensor(self.input_len[item]), torch.tensor(self.decoder_len[item]), torch.tensor(self.lang[item]), torch.tensor(self.input_ids_bert[item])



class XPersonaDataset(Dataset):  
    def __init__(self, args, tokenizer, in_max_seq=IN_MAX_SEQ, out_max_seq = OUT_MAX_SEQ, mode="train",langs = None):
        self.input_ids = []
        self.decoder_input_ids = []
        self.attention_mask = []
        self.decoder_attention_mask = []
        self.decoder_len = []
        self.input_len = []
        self.lang = [] ## since the language prediction acc is 1.0, we can use the oracle language id
        self.lang_dic = {"en":0,"fr":1,"id":2,"it":3,"jp":4,"ko":5}
        
        if mode == "train":
            if args.train_data_file == "all": self.langs = list(self.lang_dic.keys())
            else: self.langs = args.train_data_file.split(",")
        elif mode == "test":
            if args.eval_data_file == "all": self.langs = list(self.lang_dic.keys())
            else: self.langs = args.eval_data_file.split(",")
        
        if langs: self.langs = langs

        encoder_prompt_input_ids = [i for i in range(ENCODER_PROMPT_LEN)]
        for lang in self.langs:
            with open("./Xpersona_data/processed/"+mode+"_"+lang.upper()+".input", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    raw_str = line.lower() ##inform ( name = hakka restaurant ; pricerange = moderate ) & hakka restaurant is moderate -ly priced
                    if len(raw_str.split()) > in_max_seq -1: 
                        raw_str = ' '.join(raw_str.split()[:in_max_seq -1])
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_str))
                    self.input_len.append(min(len(tokenized_text),in_max_seq))
                    attention_mask = [1] *  in_max_seq
                    if len(tokenized_text) < in_max_seq: 
                        attention_mask[-(in_max_seq - len(tokenized_text)):] = [0] * (in_max_seq - len(tokenized_text))
                        tokenized_text = tokenized_text + [0] * (in_max_seq - len(tokenized_text))  ###补零
                    else:
                        tokenized_text = tokenized_text[:in_max_seq]
                    tokenized_text = encoder_prompt_input_ids + tokenized_text
                    attention_mask = [1] * ENCODER_PROMPT_LEN + attention_mask
                    self.input_ids.append(tokenized_text)
                    self.attention_mask.append(attention_mask)
                
        for lang in self.langs:
            with open("./Xpersona_data/processed/"+mode+"_"+lang.upper()+".output", encoding="utf-8") as f:
                decoder_prompt_input_ids = [i for i in range(DECODER_PROMPT_LEN)]
                for line in f:
                    line = line.strip()
                    raw_str = line.lower() 
                    if len(raw_str.split()) > out_max_seq -1: 
                        raw_str = ' '.join(raw_str.split()[:out_max_seq -1])
                    self.decoder_len.append(0)
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_str))
                    
            
                    attention_mask = [1] *  out_max_seq
                    if len(tokenized_text) < out_max_seq: 
                        attention_mask[-(out_max_seq - len(tokenized_text)):] = [0] * (out_max_seq - len(tokenized_text))
                        tokenized_text = tokenized_text + [0] * (out_max_seq - len(tokenized_text))  ###补零
                    else:
                        tokenized_text = tokenized_text[:out_max_seq]
                    self.decoder_input_ids.append(tokenized_text)
                    self.decoder_attention_mask.append(attention_mask)
                    self.lang.append(self.lang_dic[lang])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return torch.tensor(self.input_ids[item]), torch.tensor(self.attention_mask[item]), torch.tensor(self.decoder_input_ids[item]), torch.tensor(self.decoder_attention_mask[item]), torch.tensor(self.input_len[item]), torch.tensor(self.decoder_len[item]), torch.tensor(self.lang[item])

DATASET_DIC = {"intent": TextSeqDataset, "xpersona": XPersonaDataset}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def prepare_optimizer(args,model,tokenizer):
    
        optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if "prompt" in n], 'weight_decay': args.weight_decay,"lr":5e-5},
        {'params': [p for n, p in model.named_parameters() if "diction" in n], 'weight_decay': args.weight_decay,"lr":5e-5},
        {'params': [p for n, p in model.named_parameters() if "prompt" not in n and "dict" not in n], 'weight_decay': 0.0,"lr":0.0}
    ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
        if args.dictionary:
            model.bart_model.resize_token_embeddings(len(tokenizer))
        else:
            model.resize_token_embeddings(len(tokenizer))
        return optimizer,model,tokenizer




def train(args, train_dataset, model, tokenizer):  ### Train the model
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) ##1
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer,model,tokenizer = prepare_optimizer(args,model,tokenizer)

    global_step = 0; tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args) 
    print("begin")

    for epoch in train_iterator: ##EPOCH
        for step, batch in enumerate(tqdm(train_dataloader)):
            if step % 100 == 0:
                print(f"  PROGRESS: {float(global_step)/t_total*100}%")
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, _, decoder_len, lang, input_ids_bert = batch
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            decoder_input_ids = decoder_input_ids.to(args.device)
            decoder_attention_mask = decoder_attention_mask.to(args.device)
            input_ids_bert = input_ids_bert.to(args.device)

            model.train()
            if args.dictionary:
                loss,lm_logits = model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask,decoder_len = decoder_len, lang = lang, input_ids_bert = input_ids_bert)  ###inputs:[32,80], labels:[32,80]
            else:
                loss,lm_logits = model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask,decoder_len = decoder_len, lang = lang)  ###inputs:[32,80], labels:[32,80]
            loss.backward()
            
            if (step + 1) % 50 == 0:
                print(loss)
            if (step + 1) % 50 == 0:
                evaluate_dailydialog(args,model,tokenizer,generate_num=5)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()
                global_step += 1
        evaluate_dailydialog(args,model,tokenizer,suffix = "_epoch"+str(epoch))
    return global_step, tr_loss / global_step




def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--task", default=None, type=str,help="task")
    parser.add_argument("--model_type", default="mbart", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="facebook/mbart-large-cc25", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=300, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--dictionary", action='store_true',
                        help="Whether to use dictionary")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--BNM_ratio", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=5000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--text_chunk', action='store_true', help="")
    parser.add_argument('--use_reverse', action='store_true', help="")
    parser.add_argument('--with_code_loss', type=bool, default=True, help="")
    parser.add_argument('--use_tokenize', action='store_true', help="")
    parser.add_argument("--max_seq", default=80, type=int,help="")
    parser.add_argument('--mode', type=str, default=None, required = True,help="model type")
    parser.add_argument("--train", type=str, default='F')
    parser.add_argument("--test", type=str, default='F')
    args = parser.parse_args()
    args.train = args.train == "T"
    args.test = args.test == "T"
    return args

def prepare_for_main():
    args = parse_arg()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.dictionary:
        config_class, model_class, tokenizer_class = MODEL_CLASSES_DICT[args.model_type] ##<class 'transformers.models.gpt2.configuration_gpt2.GPT2Config'> <class 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'> <class 'transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer'>
    else:
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type] 

    config = config_class.from_pretrained(args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case, 
                                                cache_dir=args.cache_dir if args.cache_dir else None) ##None
    #special_tokens_dict = {'additional_special_tokens': ['__eou__']}
    #tokenizer.add_special_tokens(special_tokens_dict)

    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence) ##min(80,1024)
    if args.dictionary:
        model = model_class(config=config,args=args)
    else:
        model = model_class.from_pretrained("facebook/mbart-large-cc25",args = args, config = config)
    
    model.to(args.device)
    return args,model,tokenizer

def evaluate_dailydialog(args, model, tokenizer,suffix = "",generate_num = 100000):
    os.makedirs(args.output_dir +"/"+suffix,exist_ok=True)
    print("evaluating:",suffix)
    if args.eval_data_file == "all": langs = ["en","fr","de","es","zh","hi","ja","pt","tr"]
    else: langs = args.eval_data_file.split(",")
    
    for language in langs:
        cnt = 0
        print("language:",language)
        eval_dataset = DATASET_DIC[args.task](args,tokenizer,mode = "test",langs=[language])

        args.eval_batch_size = args.per_gpu_eval_batch_size
        if args.task == "intent": args.eval_batch_size = 1
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()
        ans = []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            if cnt > generate_num: break
            cnt += 1
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, _, decoder_len, lang, input_ids_bert = batch
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            decoder_input_ids = decoder_input_ids.to(args.device)
            decoder_attention_mask = decoder_attention_mask.to(args.device)
            input_ids_bert = input_ids_bert.to(args.device)
            with torch.no_grad():
                if args.dictionary:
                    lm_loss,_ =  model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask,lang = lang,input_ids_bert = input_ids_bert)    
                else:
                    lm_loss,_ =  model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask,lang = lang)    
                eval_loss += lm_loss.mean().item()
                nb_eval_steps += 1
                generated = decoder_input_ids[:,:decoder_len[0].item()+1]
                steps = 5
                for step in range(steps):
                    if args.dictionary:
                        lm_loss,outputs =  model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = generated,lang = lang,input_ids_bert = input_ids_bert)
                    else:
                        lm_loss,outputs =  model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = generated,lang = lang)
                    next_token_logits = outputs[:,-1, :]
                    next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                    if torch.sum(next_tokens).item() == 0: break
                    generated = torch.cat((generated, next_tokens), dim=1)
            
                out = generated.tolist()
                for i in range(len(out)):
                    text = tokenizer.decode(out[i], clean_up_tokenization_spaces=True)
                    ans.append(text)
                    if i == 0 and generate_num < 1000: print(text)
        if generate_num == 100000:
            json.dump(ans, open(args.output_dir +"/"+suffix+"/"+language+".txt",'w'), indent=2)

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))
        result = {"perplexity": perplexity}
        print(result)
    return result
        
def save_model_and_tokenizer(args,model,tokenizer,suffix = ""):
  print("Saving model checkpoint to", args.output_dir+"/"+suffix)
  if not os.path.exists(args.output_dir+"/"+suffix):
        os.makedirs(args.output_dir+"/"+suffix)
  
  model.save_pretrained(args.output_dir+"/"+suffix)
  tokenizer.save_pretrained(args.output_dir)

def load_model_and_tokenizer(args,model_class,tokenizer_class,suffix):
    print("Loading Model checkpoint.....")
    model = model_class.from_pretrained(args.output_dir+"/"+suffix)
    #tokenizer = tokenizer_class.from_pretrained(suffix, do_lower_case=args.do_lower_case)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,do_lower_case=args.do_lower_case,cache_dir=args.cache_dir if args.cache_dir else None) ##None
    model.to(args.device)
    return model,tokenizer

def main():
    args,model,tokenizer = prepare_for_main()
    if args.train:
            train_dataset = DATASET_DIC[args.task](args,tokenizer, in_max_seq=IN_MAX_SEQ, out_max_seq = OUT_MAX_SEQ,mode = "train")
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
            save_model_and_tokenizer(args,model,tokenizer)
    if args.test:
            if args.dictionary:
                config_class, model_class, tokenizer_class = MODEL_CLASSES_DICT[args.model_type] 
            else:
                config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type] 
            model, tokenizer = load_model_and_tokenizer(args,model_class,tokenizer_class,suffix = "epoch0")
            evaluate_dailydialog(args,model,tokenizer)

if __name__ == "__main__":
    main()