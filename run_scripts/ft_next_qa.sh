cd ..

export PYTHONPATH="$PYTHONPATH:$PWD"
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo $PYTHONPATH

CONFIG_PATH='config_release/next_qa.json'

horovodrun -np 1 python src/tasks/run_video_qa.py \
      --debug 1\
      --config $CONFIG_PATH \
      --output_dir /data/fukunhao/ALPRO/finetune/next_qa/$(date '+%Y%m%d%H%M%S')