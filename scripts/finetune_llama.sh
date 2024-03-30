DATA_DIR=datasets/simplicity-transfer
BATCH_SIZE=64
GRAD_ACCUM_STEPS=4
LLAMA_SIZE=7

for SIMPLICITY in simple complex
do
	python3 -m finetuning.finetune \
	--output_dir models/simplicity/Llama-2-${LLAMA_SIZE}b/finetuned_LoRA_${SIMPLICITY} \
	--model_type llama2 \
	--model_name_or_path meta-llama/Llama-2-${LLAMA_SIZE}b-hf \
    --peft_type LORA \
    --r 32 \
    --lora_alpha 16 \
	--lora_dropout 0.1 \
	--do_train \
	--num_train_epochs 1 \
	--block_size 128 \
	--save_total_limit 1 \
	--dataloader_drop_last \
	--per_device_train_batch_size $BATCH_SIZE \
	--gradient_accumulation_steps $GRAD_ACCUM_STEPS \
	--train_data_file $DATA_DIR/$SENTIMENT.txt \
    --overwrite_output_dir \
	--overwrite_cache
done