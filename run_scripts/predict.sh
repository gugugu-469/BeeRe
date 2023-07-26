echo "pid:$$"
gpu=3
cd ../

model_dir='data_dir'
data_dir='data_dir'
method_name='method name'
model_type=albert
pretrained_model_name=albert-xxlarge-v1
version1='model version gpner'
version2='model version gpner9'
version3='model version gpfilter'
# inner dim of gpner,INT NUMBER!
inner_dim_1=128
# inner dim of gpner9,INT NUMBER!
inner_dim_2=128
# inner dim of gpfilter,INT NUMBER!
inner_dim_3=128
filter_head_threshold=-8
filter_tail_threshold=-6

echo "predict gpner"
python code_main/pyrun_gpner.py --do_predict --finetuned_model_name 'gpner' --data_dir ${data_dir} --method_name ${method_name} --model_version ${version1} --devices ${gpu} --with_type --model_type ${model_type} --pretrained_model_name ${pretrained_model_name} --model_dir ${model_dir} --inner_dim ${inner_dim_1}

echo "predict gpner9"
python code_main/pyrun_gpner.py --do_predict --finetuned_model_name 'gpner9' --data_dir ${data_dir} --method_name ${method_name} --model_version ${version2} --devices ${gpu} --with_type --model_type ${model_type} --pretrained_model_name ${pretrained_model_name} --model_dir ${model_dir} --inner_dim ${inner_dim_2}

echo "result merge"
python code_main/pyrun_result_merge.py --model_version_1 ${version1} --model_version_2 ${version2} --type type1

echo "predict gpfilter"
python code_main/pyrun_gpfilter.py --do_filter --data_dir ${data_dir} --method_name ${method_name} --model_version ${version3} --model_version_1 ${version1} --model_version_2 ${version2} --devices ${gpu} --model_type ${model_type} --pretrained_model_name ${pretrained_model_name} --inner_dim ${inner_dim_3} --with_type --filter_head_threshold ${filter_head_threshold} --filter_tail_threshold ${filter_tail_threshold} --model_dir ${model_dir} 

