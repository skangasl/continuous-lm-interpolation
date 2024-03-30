LLAMA_MODEL=Llama-2-7b
MODEL_DIR=models/
PROMPTS_DATASET=datasets/prompts/writing_prompts/writing_prompts_train.jsonl
OUTPUT_DIR=generations/

for ALPHA in 0.0 0.2 0.4 0.6 0.8 1.0 0.1 0.3 0.5 0.7 0.9 -0.2 -0.4 -0.6 -0.8 -1.0 1.2 1.4 1.6 1.8 2.0 2.25 2.5 2.75 -1.25 -1.5 -1.75 -2.0 3.0 3.25 3.5 3.75 -2.25 -2.5 -2.75 -3.0 4.0
do
	python3 -m generation.generation_experiment \
	--dataset-file $PROMPTS_DATASET \
	--model meta-llama/${LLAMA_MODEL}-hf \
	--model-type weight_interpolation \
	--pos-model $MODEL_DIR/simplicity/$LLAMA_MODEL/finetuned_LoRA_simple_novel \
	--neg-model $MODEL_DIR/simplicity/$LLAMA_MODEL/finetuned_LoRA_complex_novel \
	--n 3 \
	--max-prompts 1000 \
	--max-tokens 32 \
	--batch-size 32 \
	--no-resume \
	--alpha $ALPHA \
	--use_peft True \
	$OUTPUT_DIR/simplicity/a-$ALPHA/
	
	python3 -m generation.evaluate_generations \
		--generations_file $OUTPUT_DIR/simplicity/a-$ALPHA/generations.jsonl\
		--eval_dims "simplicity"\
		--eval_model_files "models/classifiers/simplicity/pytorch_roberta.bin"\
		--num_prompts 3
done