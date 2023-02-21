python3 main_discriminator.py --dataset golf --model_name TimeGAE --epochs 100 --batch_size 32 --test_size 0.3 --cuda 0
python3 main_discriminator.py --dataset energy --model_name TimeGAE --epochs 20 --batch_size 64 --test_size 0.1 --cuda 1
python3 main_discriminator.py --dataset golf --model_name TimeGAE --epochs 20 --batch_size 64 --test_size 0.1 --cuda 2

python3 main_predictor.py --dataset golf --model_name TimeGAE --epochs 300 --batch_size 32 --cuda 3
python3 main_predictor.py --dataset energy --model_name TimeGAE --epochs 50 --batch_size 64 --cuda 4
python3 main_predictor.py --dataset winnipeg --model_name TimeGAE --epochs 30 --batch_size 64 --cuda 5
