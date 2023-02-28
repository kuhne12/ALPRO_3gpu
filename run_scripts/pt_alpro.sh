cd ..

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

CONFIG_PATH='config_release/pretrain_alpro.json'

horovodrun -np 2 python src/pretrain/run_pretrain_sparse.py \
      --debug 0\
      --config $CONFIG_PATH \
      --output_dir /data/fukunhao/ALPRO/export/vl/$(date '+%Y%m%d%H%M%S')