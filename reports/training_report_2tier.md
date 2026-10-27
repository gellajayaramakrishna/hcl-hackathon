# Router training report

Models: **small** = `qwen2.5:0.5b`, **large** = `qwen2.5:1.5b`  
Runs file: `model_runs_2tier.jsonl` (1000 runs)

## 1. Data

|                    |   test |   train |   tune |   total |
|:-------------------|-------:|--------:|-------:|--------:|
| ('math', 'easy')   |     16 |      46 |     15 |      77 |
| ('math', 'hard')   |     17 |      51 |     13 |      81 |
| ('math', 'medium') |     16 |      45 |     16 |      77 |
| ('mcq', 'easy')    |     27 |      81 |     27 |     135 |
| ('mcq', 'hard')    |     26 |      78 |     26 |     130 |
| ('total', '')      |    102 |     301 |     97 |     500 |


| source   |   n/a |   no_reuse |   reuse |   total |
|:---------|------:|-----------:|--------:|--------:|
| derived  |     0 |         20 |      30 |      50 |
| gsm8k    |   200 |          0 |       0 |     200 |
| mmlu     |   250 |          0 |       0 |     250 |
| total    |   450 |         20 |      30 |     500 |


## 2. Model runs (all 500 queries)

|                   |   n |   accuracy |   unparsed_answer |   tokens_in_avg |   tokens_out_avg |   hit_output_cap |   latency_avg_s |   latency_p50_s |   latency_p95_s |   latency_total_min |   cost_total_usd |   cost_per_query_usd |   cost_per_correct_usd |
|:------------------|----:|-----------:|------------------:|----------------:|-----------------:|-----------------:|----------------:|----------------:|----------------:|--------------------:|-----------------:|---------------------:|-----------------------:|
| ('small', 'math') | 235 |    0.35319 |           0       |         98.0979 |        263.732   |          0.06809 |         2.07099 |           1.964 |          3.7761 |             8.11137 |          0.0147  |              6e-05   |                0.00018 |
| ('small', 'mcq')  | 265 |    0.36226 |           0       |        147.709  |          4.37358 |          0       |         0.21397 |           0.196 |          0.3336 |             0.94502 |          0.00415 |              2e-05   |                4e-05   |
| ('large', 'math') | 235 |    0.70213 |           0       |         98.0979 |        173.055   |          0.00851 |         3.25421 |           3.059 |          5.5745 |            12.7457  |          0.03132 |              0.00013 |                0.00019 |
| ('large', 'mcq')  | 265 |    0.52075 |           0.00377 |        147.709  |          3.78868 |          0       |         0.4666  |           0.361 |          0.863  |             2.06082 |          0.01235 |              5e-05   |                9e-05   |


Per model, both task types together:

| tier   |   accuracy |   latency_total_min |   cost_total_usd |
|:-------|-----------:|--------------------:|-----------------:|
| small  |      0.358 |             9.05638 |          0.01885 |
| large  |      0.606 |            14.8065  |          0.04366 |


Total labelling time: **23.9 min** for 1000 calls; total (placeholder) cost: **$0.0625**.


Accuracy by proxy difficulty:

|                    |   small |   large |
|:-------------------|--------:|--------:|
| ('math', 'easy')   |   0.468 |   0.753 |
| ('math', 'hard')   |   0.284 |   0.605 |
| ('math', 'medium') |   0.312 |   0.753 |
| ('mcq', 'easy')    |   0.415 |   0.711 |
| ('mcq', 'hard')    |   0.308 |   0.323 |


Who gets each query right:

|                           |   queries |
|:--------------------------|----------:|
| small=wrong / large=right |       162 |
| small=wrong / large=wrong |       159 |
| small=right / large=right |       141 |
| small=right / large=wrong |        38 |


## 3. Classifier quality (AUC: 0.5 = random, 1.0 = perfect)

|       |   small |   large |
|:------|--------:|--------:|
| train |   0.972 |   0.927 |
| tune  |   0.654 |   0.529 |
| test  |   0.588 |   0.793 |


A big drop from train to tune/test means overfitting (memorising the training queries).


## 4. Router vs baselines (test split, 102 queries)

Threshold 0.45 (chosen on tune), score = accuracy - 0.5 x cost/cost(always largest). Router decision time: 0.12 ms per query (not included in latency below).


|              |   accuracy |   acc_mcq |   acc_math |   cost_usd |   cost_vs_always_large |   latency_total_s |   latency_avg_s |   score |   share_small |   share_large |
|:-------------|-----------:|----------:|-----------:|-----------:|-----------------------:|------------------:|----------------:|--------:|--------------:|--------------:|
| always_small |     0.3333 |    0.4151 |     0.2449 |     0.0042 |                 0.4438 |           119.089 |          1.1675 |  0.1114 |        1      |        0      |
| always_large |     0.5784 |    0.4906 |     0.6735 |     0.0094 |                 1      |           201.565 |          1.9761 |  0.0784 |        0      |        1      |
| oracle       |     0.6765 |    0.6604 |     0.6939 |     0.0061 |                 0.6429 |           148.656 |          1.4574 |  0.355  |        0.6569 |        0.3431 |
| router       |     0.451  |    0.4151 |     0.4898 |     0.0072 |                 0.7609 |           155.56  |          1.5251 |  0.0705 |        0.4412 |        0.5588 |
