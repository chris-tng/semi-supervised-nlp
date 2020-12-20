# Semi Supervised NLP

Implementation of semi-supervised learning techniques: UDA, MixMatch, Mean-teacher, focusing on NLP.

Notes:
- Instead of `mixup` in the original paper, I use [Manifold Mixup](https://arxiv.org/abs/1806.05236) , which is better suited for NLP application.

## Encoder

- Any `encoder` can be used: transformer, LSTM, etc. The default is [LSTMWeightDrop](https://github.com/chris-tng/semi-supervised-nlp/blob/8148d07f79a24acdd23e4ae55b1c12b7cf2ae7b7/layers.py#L130), used in [AWD-LSTM](https://arxiv.org/pdf/1708.02182.pdf), inspired by `fast.ai`-v1. 

- Since this repo is mainly concerned with exploring SSL techniques, using Transformer can be overkill. It could dominate the progress made by SSL, not to mention long training time.

## Data Augmentation

There're many data augmentation techniques in Computer Vision, not so much in NLP. It's an open research into strong data augmentation in NLP. So far, what I found effectively is `back-translation`, confirmed by UDA paper. There're many ways to perform back-translation, one simple way is to use [MarianMT](https://huggingface.co/transformers/model_doc/marian.html), shipped in the excellent `huggingface-transformers`.

- Some data augmentation techniques I would like to explore

- [ ] TF-IDF word replacement
- [ ] Sentence permutation
- [ ] Nearest neighbor sentence replacement

## Citations

```
@article{xie2019unsupervised,
  title={Unsupervised Data Augmentation for Consistency Training},
  author={Xie, Qizhe and Dai, Zihang and Hovy, Eduard and Luong, Minh-Thang and Le, Quoc V},
  journal={arXiv preprint arXiv:1904.12848},
  year={2019}
}

@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```

