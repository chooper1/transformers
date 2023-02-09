export CUDA_VISIBLE_DEVICES=''
python run_translation.py \
    --model_name_or_path google/mt5-small \
    --do_eval \
    --source_lang en \
    --target_lang de \
    --dataset_name iwslt2017 \
    --dataset_config_name iwslt2017-de-en \
    --output_dir /scratch/chooper/iwslt/tst-translation2 \
    --source_prefix 'translate English to German: ' \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --overwrite_output_dir \
    --cache_dir /scratch/chooper/iwslt/cache-translation2 \
    --predict_with_generate
