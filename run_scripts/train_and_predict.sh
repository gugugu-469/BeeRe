echo "pid:$$"
gpu=2
cd ../

data_dir=./ACE05-DyGIE/processed_data
method_name=ace05
model_type=bert
pretrained_model_name=bert-base-uncased
version1=$(command date +%m-%d-%H-%M)

# TRAIN
echo "version1:${version1}"
python -u code_main/pyrun_gpner.py --do_train --finetuned_model_name 'gpner' --epochs 2 --data_dir ${data_dir} --method_name ${method_name} --model_version ${version1}  --devices ${gpu} --with_type --model_type ${model_type} --pretrained_model_name ${pretrained_model_name}

version2=$(command date +%m-%d-%H-%M)
echo "version2:${version2}" 
python -u code_main/pyrun_gpner.py --do_train --finetuned_model_name 'gpner9' --epochs 2 --data_dir ${data_dir} --method_name ${method_name} --model_version ${version2}  --devices ${gpu} --with_type --model_type ${model_type} --pretrained_model_name ${pretrained_model_name}

version3=$(command date +%m-%d-%H-%M)
echo "version3:${version3}"
python -u code_main/pyrun_gpfilter.py --do_train --epochs 2 --data_dir ${data_dir} --method_name ${method_name} --model_version ${version3}  --devices ${gpu} --with_type --model_type ${model_type} --pretrained_model_name ${pretrained_model_name}

# PREDICT
echo "predict gpner"
python code_main/pyrun_gpner.py --do_predict --finetuned_model_name 'gpner' --data_dir ${data_dir} --method_name ${method_name} --model_version ${version1} --devices ${gpu} --with_type --model_type ${model_type} --pretrained_model_name ${pretrained_model_name}

echo "predict gpner9"
python code_main/pyrun_gpner.py --do_predict --finetuned_model_name 'gpner9' --data_dir ${data_dir} --method_name ${method_name} --model_version ${version2} --devices ${gpu} --with_type --model_type ${model_type} --pretrained_model_name ${pretrained_model_name}
 
echo "result merge"
python code_main/pyrun_result_merge.py --model_version_1 ${version1} --model_version_2 ${version2} --type type1

echo "predict gpfilter"
python code_main/pyrun_gpfilter.py --do_filter --data_dir ${data_dir} --method_name ${method_name} --model_version ${version3} --model_version_1 ${version1} --model_version_2 ${version2} --devices ${gpu} --with_type --model_type ${model_type} --pretrained_model_name ${pretrained_model_name}
