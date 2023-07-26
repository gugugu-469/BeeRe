# BeeRe

BeeRe(A Bidirectional Extraction-then-Evaluation Framework for Complex Relation Extraction)

### folders and files:



`/dataset` contains all the datasets used in this paper, and we provide formatted samples for each dataset.

`/checkpoint` is used to store model parameters.

`/code_main` contains three methods' main code in BeeRe

`/codes` contains the auxiliary code required for the code in `/code_main`


`/result_output` is used to store model prediction result.

`/run_scripts` contains the script which runs the code in `/code_main`


### Train

```
cd ./run_sciprts
bash train_amd_predict.sh
```