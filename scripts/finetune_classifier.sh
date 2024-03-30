DATASET="datasets/formality-transfer/formality_test.csv"
OUTPUT_DIR=models/classifiers/formality

python3 -m finetuning.finetune_classifier \
	--dataset-file $DATASET \
	--csv True \
	--loss-function BCE \
	--n-epoch 3 \
	$OUTPUT_DIR