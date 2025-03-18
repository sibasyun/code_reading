# faster R-CNN

[torchvisionの実装](https://pytorch.org/vision/stable/models/faster_rcnn.html)  

参考リンク  
[物体検出Faster R-CNN （Faster Region-based CNN）](https://qiita.com/DeepTama/items/0cb9ca2d35c200deed37)  
[torchvisionの実装から見るFaster R-CNN](https://qiita.com/ground0state/items/08be7707069bdba10)  
[R-CNNの系譜をたどる](https://zenn.dev/kabupen/articles/note-20230320-00)  
[RCNNで使われるSelective Searchについてまとめてみる](https://blog.shikoan.com/selective-search-rcnn/)

## R-CNNモデルの概要
### R-CNN
- Regional CNNの略称で、初出は2013年。
- 以下のような流れで処理を行う。
  - Region of Interest(ROI)をselective searchと呼ばれる手法で抽出する。
  - 抽出したRoIを227×227に変換する。
  - CNNに入力して特徴量を計算する。
  - 計算された特徴量をSVMに入力し、クラス分類タスクを解く。

### Fast R-CNN
- CNNへの入力を、生画像ではなく一度CNNを通した特徴マップとすることで計算量の削減を図った手法。

### Faster R-CNN
- RoIの抽出部分及びクラス分類部分もCNNに変更することで、より高速な処理を実現。
- 以下のような流れで処理を行う。
  - ResNetやVGG16などのCNNで特徴マップを生成する
  - Region Proposal Network(RPN)で、RoIを生成する
    - RPNでは、固定サイズのアンカーを生成し、各アンカーをどれだけ変形させれば物体が映っているBounding Boxになるかを予測する
  - RoI Pooling層で、異なるサイズのProposalを固定サイズの特徴マップに変換する
  - 全結合層とクラス分類器を通して、クラスラベル及びBounding Boxの座標を予測する

## torchvisionにおけるFaster R-CNNの実装
[faster_rcnn](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py)の実装はここから確認できる。  
`GeneralizedRCNN`クラスを継承していることがわかる。  

コンストラクタ部分の実装。
- backbone : 特徴マップの抽出に用いるCNNを渡す。
- num_classes : 学習データセットのクラス数。
- rpn_anchor_generator : アンカーの生成器。Noneとした場合はコンストラクタ内で生成される。
- rpn_head : RPNのモジュール。Noneとした場合はコンストラクタ内で生成される。
- box_head : クラス分類器。Noneとした場合はコンストラクタ内で生成される。
- box_predictor : bounding boxの推測を行う。Noneとした場合貼コンストラクタ内で生成される。

backboneだけは明示的に渡さなくてはならず、他のモジュールはデフォルトの設定で生成される。  

```python
def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        **kwargs,
    ):
```

コンストラクタ内での各モジュールの生成について見ていく。  

