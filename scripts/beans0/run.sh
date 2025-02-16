#python prepare_beans0.py --output_dir "gs://foundation-model-data/beans0/0.0.1" --components_to_process lifestage call-type --batch_size 300 --replace_16k --versions "0.0.1" --use_local_cache

python prepare_beans0_parallel.py --output_dir "gs://foundation-model-data/beans0/raw/" --components_to_process gibbons lifestage captioning call-type unseen-species-sci unseen-species-cmn unseen-species-tax unseen-genus-cmn unseen-genus-sci unseen-genus-tax zf-indiv --batch_size 300 --replace_16k --versions "0.1.0"
