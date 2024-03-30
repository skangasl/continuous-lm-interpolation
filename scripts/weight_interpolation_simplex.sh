LLAMA_MODEL=Llama-2-7b
MODEL_DIR=models/
CLASSIFIER_DIR=models/classifiers
PROMPTS_DATASET=datasets/prompts/writing_prompts/writing_prompts_train.jsonl
OUTPUT_DIR=generations/
DIMS=("humor" "formality" "simplicity")
MODEL_DIRS=("${CLASSIFIER_DIR}/humor/pytorch_roberta.bin" "${CLASSIFIER_DIR}/formality/pytorch_roberta.bin" "${CLASSIFIER_DIR}/simplicity/pytorch_roberta.bin")
LAMBDA_SUM=1.0

for LAMBDA1 in 0.0 0.2 0.4 0.6 0.8 1.0
do
	for LAMBDA2 in 0.0 0.2 0.4 0.6 0.8 1.0
	do
		for LAMBDA3 in 0.0 0.2 0.4 0.6 0.8 1.0
		do
			CURR_LAMBDA=$(python -c "print(int(round($LAMBDA1+$LAMBDA2+$LAMBDA3, 2) == $LAMBDA_SUM))")

			if [ "$CURR_LAMBDA" -eq 1 ]; then
				for ALPHA1 in 1.0
				do
					for ALPHA2 in 1.0
					do
						for ALPHA3 in 1.0
						do
							OUTFILE="configs/sentiment-politeness-formality-a-$ALPHA1-$ALPHA2-$ALPHA3-l-$LAMBDA1-$LAMBDA2-$LAMBDA3.json"
							ALPHAS=("${ALPHA1}" "${ALPHA2}" "${ALPHA3}")
							LAMBDAS=("${LAMBDA1}" "${LAMBDA2}" "${LAMBDA3}")

							export ALPHAS
							export LAMBDAS

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
								--batch-size 32 \
								--no-resume \
								--interpolation-parameter-dict-path $OUTFILE \
								--use_peft True \
								$OUTPUT_DIR/sentiment-politeness-formality-a-$ALPHA1-$ALPHA2-$ALPHA3-l-$LAMBDA1-$LAMBDA2-$LAMBDA3/
							
							python3 -m generation.evaluate_generations \
								--generations_file $OUTPUT_DIR/humor-formality-simplicity-a-$ALPHA1-$ALPHA2-$ALPHA3-l-$LAMBDA1-$LAMBDA2-$LAMBDA3/generations.jsonl\
								--eval_dims "${DIMS[@]}"\
								--eval_model_files "${MODEL_DIRS[@]}"\
								--num_prompts 3
						done
					done
				done
			fi
		done
	done
done