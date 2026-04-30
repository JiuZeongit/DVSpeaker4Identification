# DVSpeaker4Identification

A public, streamlined training and evaluation codebase for **close-set speaker identification on DVSpeaker**.

## Dataset format

The code expects the [DVSpeaker](https://github.com/JiuZeongit/NeuroLip) dataset to be organized as:

```text
DVSpeaker/
├── 1/
│   ├── 1_0_0_xxx.npy
│   ├── 1_45_3_xxx.npy
│   └── ...
├── 2/
│   └── ...
└── ...
```

- Each folder name is a person ID.
- Each `.npy` filename must start with:

```text
{light}_{degree}_{num}_...
```

where:
- `light` is `0` or `1`
- `degree` is `0`, `45`, or `90`
- `num` is a digit used for optional filtering

Each `.npy` file should contain event data with fields `(x, y, p, t)`.
Structured arrays and plain `N x 4` arrays are both supported.

## Environment

### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

The training protocol is:
- `train_ids` are used for supervised training with cross-entropy classification
- `test_ids` are used for close-set identification evaluation
- in each evaluation condition, each test identity contributes `--enroll-per-id` samples to the gallery
- the remaining samples of the same identity form the probe set
- the best checkpoint is selected by the average Rank-1 across all requested evaluation conditions

Example:

```bash
python train.py   --data-root /path/to/DVSpeaker   --train-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30   --test-ids 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50   --train-lights 0,1   --train-degrees 0,45,90   --batch-size 16   --epochs 50   --lr 1e-3   --num-workers 4   --enroll-per-id 3   --sensor-size 200,160   --num-frames 32   --alpha 4   --save-root result_dvspeaker
```

Saved files:
- `last.pt`
- `best_avg_rank1.pt`
- `best_metrics.json`
- `history.json`

## Standard close-set evaluation

This reproduces the training-time protocol using the saved checkpoint:

```bash
python eval.py   --data-root /path/to/DVSpeaker   --ckpt result_dvspeaker/best_avg_rank1.pt   --test-ids 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50   --eval-conditions 0:0,1:0,1:45,1:90   --enroll-per-id 3   --save-json result_eval/standard_close_set.json
```

## Cross-condition evaluation

This script allows you to explicitly control gallery and probe conditions.
For example, use `light=1, degree=0` as gallery and evaluate on four probe conditions.

### Same-condition: L1D0 → L1D0

```bash
python eval_cross_condition.py   --data-root /path/to/DVSpeaker   --ckpt result_dvspeaker/best_avg_rank1.pt   --test-ids 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50   --gallery-lights 1 --gallery-degrees 0   --probe-lights 1 --probe-degrees 0   --enroll-per-id 3   --save-json result_eval/L1D0_to_L1D0.json
```

### Cross-condition: L1D0 → L1D45

```bash
python eval_cross_condition.py   --data-root /path/to/DVSpeaker   --ckpt result_dvspeaker/best_avg_rank1.pt   --test-ids 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50   --gallery-lights 1 --gallery-degrees 0   --probe-lights 1 --probe-degrees 45   --enroll-per-id 3   --save-json result_eval/L1D0_to_L1D45.json
```

### Cross-condition: L1D0 → L1D90

```bash
python eval_cross_condition.py   --data-root /path/to/DVSpeaker   --ckpt result_dvspeaker/best_avg_rank1.pt   --test-ids 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50   --gallery-lights 1 --gallery-degrees 0   --probe-lights 1 --probe-degrees 90   --enroll-per-id 3   --save-json result_eval/L1D0_to_L1D90.json
```

### Cross-condition: L1D0 → L0D0

```bash
python eval_cross_condition.py   --data-root /path/to/DVSpeaker   --ckpt result_dvspeaker/best_avg_rank1.pt   --test-ids 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50   --gallery-lights 1 --gallery-degrees 0   --probe-lights 0 --probe-degrees 0   --enroll-per-id 3   --save-json result_eval/L1D0_to_L0D0.json
```

## Notes

- The default input representation is a 2-channel event-count volume with optional `log1p` compression.
- The Slow pathway is built by temporal downsampling controlled by `--alpha`.
- Close-set identification is implemented with nearest-neighbor search on L2 distance over normalized embeddings.
- Training and test identities must be disjoint.


## Citation

If you find our work useful in your research, please cite:


```
@misc{ 
}
```

```
@misc{yao2026neurolipeventdrivenspatiotemporallearning,
      title={NeuroLip: An Event-driven Spatiotemporal Learning Framework for Cross-Scene Lip-Motion-based Visual Speaker Recognition}, 
      author={Junguang Yao and Wenye Liu and Stjepan Picek and Yue Zheng},
      year={2026},
      eprint={2604.15718},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.15718}, 
}
```