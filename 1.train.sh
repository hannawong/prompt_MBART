CUDA_VISIBLE_DEVICES="5" \
python train.py --output_dir=./outputs/mbart_translate_100 \
--model_name_or_path=facebook/mbart-large-cc25 --do_train  \
--do_eval \
--eval_data_file=all \
--learning_rate 0.00625 --use_tokenize \
--overwrite_cache \
--train_data_file=all \
--overwrite_output_dir \
--block_size=80 \
--seed 0 \
--mode=adapter --gradient_accumulation_step=1 --num_train_epochs 10 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 1 \
--train T --test T \
--task translate --prompt_lang_specific None
