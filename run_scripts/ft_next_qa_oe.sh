cd ..

export PYTHONPATH="$PYTHONPATH:$PWD"
export CUDA_VISIBLE_DEVICES=0,1,2
#echo $PYTHONPATH

CONFIG_PATH='config_release/next_qa_oe.json'

horovodrun -np 3 python src/tasks/run_video_qa.py \
      --debug 0\
      --config $CONFIG_PATH \
      --output_dir /data3/fukunhao/ALPRO/finetune/next_qa_oe/$(date '+%Y%m%d%H%M%S')
