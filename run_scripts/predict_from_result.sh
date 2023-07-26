echo "pid:$$"
gpu=3
cd ../

model_dir='data_dir'
data_dir='data_dir'
method_name='method name'
model_type=albert
pretrained_model_name=albert-xxlarge-v1
version='model version'
# inner dim,INT NUMBER!!
inner_dim=128
filter_head_thresholds=-8
filter_tail_thresholds=-6

echo 'predict from result'
python -u code_main/pyrun_gpner.py \
    --do_predict_from_result \
    --result_path 'set result path'  \
    --finetuned_model_name 'gpner9' \
    --data_dir ${data_dir} \
    --method_name ${method_name} \
    --model_version ${version} \
    --devices ${gpu} \
    --with_type \
    --model_type ${model_type} \
    --pretrained_model_name ${pretrained_model_name} \
    --model_dir ${model_dir} \
    --inner_dim ${inner_dim}
