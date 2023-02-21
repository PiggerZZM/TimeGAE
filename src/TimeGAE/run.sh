python3 main_TimeGAE.py --dataset golf --batch_size 16 --epochs 3000 --emb_size 16 --feat_extract transformer --max_norm 1 --num_layers 1 --num_heads 1 --cuda 0
python3 main_TimeGAE.py --dataset energy --batch_size 256 --epochs 1000 --emb_size 16 --feat_extract transformer --max_norm 1 --num_layers 1 --num_heads 1 --cuda 1
python3 main_TimeGAE.py --dataset winnipeg --batch_size 64 --epochs 1000 --emb_size 16 --feat_extract transformer --max_norm 0.1 --num_layers 1 --num_heads 1 --cuda 2


python3 main_TimeGVAE.py --dataset golf --batch_size 16 --epochs 1000 --emb_size 16 --feat_extract transformer --max_norm 0.1 --num_layers 1 --num_heads 1 --scheduler --cuda 0
python3 main_TimeGVAE.py --dataset energy --batch_size 256 --epochs 300 --emb_size 16 --feat_extract transformer --max_norm 1 --num_layers 2 --num_heads 2 --scheduler --cuda 1
python3 main_TimeGVAE.py --dataset winnipeg --batch_size 64 --epochs 300 --emb_size 16 --feat_extract transformer --max_norm 0.1 --num_layers 2 --num_heads 2 --scheduler --cuda 2
