LLAMA_MODEL=Llama-2-7b
MODEL_DIR=models/
CLASSIFIER_DIR=models/classifiers
PROMPTS_DATASET=datasets/prompts/writing_prompts/writing_prompts_train.jsonl
OUTPUT_DIR=generations/5dim
DIMS=("sentiment" "politeness" "humor" "formality" "simplicity")
MODEL_DIRS=("${CLASSIFIER_DIR}/sentiment/pytorch_roberta.bin" "${CLASSIFIER_DIR}/politeness/pytorch_roberta.bin" "${CLASSIFIER_DIR}/humor/pytorch_roberta.bin" "${CLASSIFIER_DIR}/formality/pytorch_roberta.bin" "${CLASSIFIER_DIR}/simplicity/pytorch_roberta.bin")

DIM_NUM=0
CURR_DIM=${DIMS[$DIM_NUM]}
BASE_ALPHA=1.0

for BL in 0.05 0.15 0.2
do
	LAMBDAS=("${BL}" "${BL}" "${BL}" "${BL}" "${BL}")
	LAMBDAS[$DIM_NUM]=$(python -c "print(round(1.0-4*$BL, 1))")

	export LAMBDAS

	for ALPHA in 0.0 0.2 0.4 0.6 0.8 1.0
	do
		OUTFILE="configs/5dim-$CURR_DIM-a-$ALPHA-${BASE_ALPHA}-l-${BL}.json"
		ALPHAS=("${BASE_ALPHA}" "${BASE_ALPHA}" "${BASE_ALPHA}" "${BASE_ALPHA}" "${BASE_ALPHA}")
		ALPHAS[$DIM_NUM]="${ALPHA}"

		export ALPHAS

		python3 scripts/create_json_file.py \
			-d "${DIMS[@]}"\
			-a "${ALPHAS[@]}"\
			-l "${LAMBDAS[@]}"\
			-o "${OUTFILE}"

		python3 -m generation.generation_experiment \
			--dataset-file $PROMPTS_DATASET \
			--model-type weight_interpolation_multidim \
			--model meta-llama/${LLAMA_MODEL}-hf \
			--expert-model-dict-path "configs/lora_llama_config.json" \
			--n 3 \
			--max-prompts 1000 \
			--max-tokens 32 \
			--batch-size 64 \
			--no-resume \
			--interpolation-parameter-dict-path $OUTFILE \
			--use_peft True \
			$OUTPUT_DIR/5dim-$CURR_DIM-a-$ALPHA-${BASE_ALPHA}-l-${BL}/
		
		python3 -m generation.evaluate_generations \
			--generations_file $OUTPUT_DIR/5dim-$CURR_DIM-a-$ALPHA-${BASE_ALPHA}-l-${BL}/generations.jsonl\
			--eval_dims "${DIMS[@]}"\
			--eval_model_files "${MODEL_DIRS[@]}"\
			--num_prompts 3
	done
done