export CUDA_VISIBLE_DEVICES=7

# For example
for id in `seq 0 1 10`
do
    python train.py --domain res --train-op 0 --use-doc 1 --use-opinion 0 --senti-layers 1 --doc-senti-layers 0 --doc-domain-layers 0 --interactions 2 --which_dual dual1 --pre-epochs 5 --aspect-layers 2 --opinion-layers 2 --cap_dim 200 --use-bert-cls 5 --use-bert 1
done
