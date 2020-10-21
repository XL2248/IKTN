export CUDA_VISIBLE_DEVICES=7
batch_size=(16 25 32 50 64 80 100)
learning_rate=(0.0001 0.0002 0.0005 0.0008)
# 0.0002 0.0005 0.0008)
for idx in `seq 1 1 2`
do
    for id in `seq 0 1 10`
    do
        python train.py --domain res --train-op 0 --use-doc 1 --use-opinion 0 --senti-layers 1 --doc-senti-layers 0 --doc-domain-layers 0 --interactions $idx --which_dual dual1 --pre-epochs 5 --aspect-layers 2 --opinion-layers 2 --cap_dim 200 --use-bert-cls 5 --use-bert 1
    done
done
#python train.py --domain lt --train-op 0 --use-doc 1 --use-opinion 0 --shared-layers 2 --senti-layers 0 --which_dual dual1
#python train.py --domain res_15 --train-op 0 --use-doc 1 --use-opinion 0 --shared-layers 2 --senti-layers 0 --which_dual dual1
