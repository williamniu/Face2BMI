# Evaluation Summary

Deployed model: ensemble (facenet_casia, facenet_vggface2, vgg16_imagenet)
Overall Pearson r: 0.678
Male Pearson r: 0.693
Female Pearson r: 0.668
MAE: 5.028
RMSE: 7.154
Pairwise accuracy: 0.7433333333333333

## Comparison vs paper

| Model | Overall r | Male r | Female r |
|---|---:|---:|---:|
| Paper VGG-Net | 0.47 | 0.58 | 0.36 |
| Paper VGG-Face | 0.65 | 0.71 | 0.57 |
| **This run (ensemble)** | **0.678** | **0.693** | **0.668** |

Beats paper VGG-Face overall? **True**