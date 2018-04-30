GPUID=$1
MODELPATH="$(python -m entitylinking.mlearning.training.train_entity_linker configs/trainmodel_vector_model.yaml 1 $GPUID | tail -n1)"
python -m entitylinking.evaluation.evaluate $MODELPATH configs/evaluate_webqsp.yaml
python -m entitylinking.evaluation.evaluate $MODELPATH configs/evaluate_graphquestions.yaml

