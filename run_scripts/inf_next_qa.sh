cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

STEP='34800'

CONFIG_PATH='config_release/next_qa.json'

TXT_DB='/data/fukunhao/NExT-QA/nextqa_mc/val.csv'
IMG_DB='/data/fukunhao/VidOR/video'

horovodrun -np 4 python src/tasks/run_video_qa.py \
      --debug 1\
      --do_inference 1 \
      --inference_split val \
      --inference_model_step $STEP \
      --inference_txt_db $TXT_DB \
      --inference_img_db $IMG_DB \
      --inference_batch_size 16 \
      --output_dir /data/fukunhao/ALPRO/finetune/next_qa/20221115113941 \
      --config $CONFIG_PATH

#python src/tasks/run_video_qa.py \
#      --do_inference 1 \
#      --inference_split test \
#      --inference_model_step $STEP \
#      --inference_txt_db $TXT_DB \
#      --inference_img_db $IMG_DB \
#      --inference_batch_size 64 \
#      --output_dir /data/fukunhao/ALPRO/finetune/next_qa/20221112105822/ \
#      --config $CONFIG_PATH