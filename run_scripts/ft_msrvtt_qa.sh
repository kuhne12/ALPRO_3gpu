cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
export CUDA_VISIBLE_DEVICES=3,4
echo $PYTHONPATH

CONFIG_PATH='config_release/msrvtt_qa.json'

horovodrun -np 1 python src/tasks/run_video_qa.py \
      --debug 1\
      --config $CONFIG_PATH \
      --output_dir /data/fukunhao/ALPRO/finetune/msrvtt_qa/$(date '+%Y%m%d%H%M%S')
