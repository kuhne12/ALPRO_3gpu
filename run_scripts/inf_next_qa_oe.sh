cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2

STEP='15510'

CONFIG_PATH='config_release/next_qa_oe.json'

TXT_DB='/data3/fukunhao/NExT-QA/nextqa_oe/test.csv'
IMG_DB='/data3/fukunhao/VidOR/video'

horovodrun -np 3 python src/tasks/run_video_qa.py \
      --do_inference 1 \
      --inference_split test \
      --inference_model_step $STEP \
      --inference_txt_db $TXT_DB \
      --inference_img_db $IMG_DB \
      --inference_batch_size 64 \
      --output_dir /data3/fukunhao/ALPRO/finetune/next_qa_oe/20230226102026/ \
      --config $CONFIG_PATH \
      --task next_qa_oe

#python src/tasks/run_video_qa.py \
#      --do_inference 1 \
#      --inference_split test \
#      --inference_model_step $STEP \
#      --inference_txt_db $TXT_DB \
#      --inference_img_db $IMG_DB \
#      --inference_batch_size 64 \
#      --output_dir /data/fukunhao/ALPRO/finetune/next_qa/20221112105822/ \
#      --config $CONFIG_PATH