export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PADDLE_TRAINER_ID=0
export PADDLE_TRAINERS_NUM=1
rm -rf log

python resnet_pipeline.py
