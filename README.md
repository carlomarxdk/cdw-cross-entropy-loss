# Class Distance Weighted Cross-Entropy Loss
Implementation of the Class Distance Weighted Cross-Entropy Loss in PyTorch.

The CDW Cross-Entropy Loss is presented in the [Class Distance Weighted Cross-Entropy Loss for Ulcerative Colitis Severity Estimation](https://arxiv.org/abs/2202.05167) (see citations) and further extended in the [Using sequences of life-events to predict human lives](https://www.nature.com/articles/s43588-023-00573-5).

This repository provides a simple implementation of *original* CDW Cross-Entropy and the one used in the [life2vec](https://github.com/SocialComplexityLab/life2vec) case.

## How to use?

```python
from cdw_cross_entropy_loss import CDW_CELoss
loss = CDW_CELoss(num_classes = 4, 
                 alpha = 2., # Weight or penalty term
                 delta  = 3., # Only used for the Huber Transform
                 reduction  = "mean",
                 transform  = "log",  # Original paper uses power transform
                 eps = 1e-8)
```

## Citations

```bibtex
@inproceedings{polat2022class,
  title={Class distance weighted cross-entropy loss for ulcerative colitis severity estimation},
  author={Polat, Gorkem and Ergenc, Ilkay and Kani, Haluk Tarik and Alahdab, Yesim Ozen and Atug, Ozlen and Temizel, Alptekin},
  booktitle={Annual Conference on Medical Image Understanding and Analysis},
  pages={157--171},
  year={2022},
  organization={Springer}
}
```

```bibtex
@article{savcisens2024using,
      author={Savcisens, Germans and Eliassi-Rad, Tina and Hansen, Lars Kai and Mortensen, Laust Hvas and Lilleholt, Lau and Rogers, Anna and Zettler, Ingo and Lehmann, Sune},
      title={Using sequences of life-events to predict human lives},
      journal={Nature Computational Science},
      year={2024},
      month={Jan},
      day={01},
      volume={4},
      number={1},
      pages={43-56},
      issn={2662-8457},
      doi={10.1038/s43588-023-00573-5},
      url={https://doi.org/10.1038/s43588-023-00573-5}
}
```

```bibtex
@misc{life2vec_code,
  author = {Germans Savcisens},
  title = {Official code for the "Using Sequences of Life-events to Predict Human Lives" paper},
  note = {GitHub: SocialComplexityLab/life2vec},
  year = {2023},
  howpublished = {\url{https://doi.org/10.5281/zenodo.10118621}},
}
```
