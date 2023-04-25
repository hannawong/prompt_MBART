CUDA_VISIBLE_DEVICES="5" \
python train.py --output_dir=./outputs/prompt_intent_dict \
--model_name_or_path=facebook/mbart-large-cc25 --do_train  \
--do_eval \
--eval_data_file=en \
--learning_rate 0.00625 --use_tokenize \
--overwrite_cache \
--train_data_file=en \
--overwrite_output_dir \
--block_size=80 \
--seed 0 \
--mode=adapter --gradient_accumulation_step=1 \
--num_train_epochs 5 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 8 \
--train T --test F \
--task intent \
--dictionary


####prompt: 0.88

#### finetune: 1.3689 -> 1.3085 -> 1.0186 -> 1.0042 -> 1.0030   1.0063