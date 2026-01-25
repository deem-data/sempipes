model="gemini"
search="mct_search"

for pipeline in midwestsurvey fraudbaskets churn traffic tweets; do
for seed in 0 42 748 1985 18102022; do
CUDA_VISIBLE_DEVICES=0 poetry run python -m experiments.colopro.minibench $pipeline $model $search $seed 2>&1 |tee ${pipeline}_${model}_${search}_${seed}.txt
done
done``