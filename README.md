
<h1 align='center'>Self-Similarity Priors: <br> Neural Collages as Differentiable  Fractal Representations
</h1>

<div align="center">

https://user-images.githubusercontent.com/34561392/204065084-e3c80d70-8bbb-4ac4-9449-b64dddb85fcc.mp4


[![License](https://img.shields.io/badge/License-MIT-black.svg?)](https://papers.nips.cc/paper/2020/hash/f1686b4badcf28d33ed632036c7ab0b8-Abstract.html)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2022-red.svg?)]()
[![License](https://img.shields.io/badge/License-MIT-black.svg?)]()
[![arXiv](https://img.shields.io/badge/arXiv-2204.07673-purple.svg?)](https://arxiv.org/abs/2204.07673)
[![Blog](https://img.shields.io/badge/Blog-blue.svg?)](https://zymrael.github.io/self-similarity-prior/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-orange.svg?)](https://huggingface.co/spaces/Zymrael/Neural-Collage-Fractalization)

</div>


Many patterns in nature exhibit self-similarity: they can be compactly described via self-referential transformations. 

In this work, we investigate the role of learning in the automated discovery of self-similarity and in its utilization for downstream tasks. We design a novel class of implicit operators, Neural Collages, which (1) represent data as the parameters of a self-referential, structured transformation, and (2) employ hypernetworks to amortize the cost of finding these parameters to a single forward pass. 

We investigate how to leverage the representations produced by Neural Collages in various tasks:

* Lossy image compression
* Deep generative modeling
* Image fractalization

## The Upshot

We introduce a contractive operator as a layer. One application of a Collage Operator involves
tokenizing the input domain into two partitions: sources and targets. The source tokens are combined into target tokens, which are then "stitched together" (as a collage). The parameters of a Collage operator relate parts of the input to itself, and can thus be used to achieve high rates of compression in self-similar data. 

The machinery behind collage operators is inspired by fractal compression schemes. We provide a self-contained differentiable implementation of a simple fractal compression scheme in `jax_src/compress/fractal.py`. The script
`scripts/fractal_compress_img.py` can be used to fractal compress a batch of images using this method. 

## Codebase

The codebase is organized as follows. We provide a simple implementation of a Collage Operator and related utilities under `torch_src/`. The bulk of the experiments has been carried out in `jax`. Under `scripts/` we provide training and evalaution scripts for the three main experiments. The lossy image compression experiment is performed on a slice of the aerial dataset described in the paper, which can be found at this
[link](https://captain-whu.github.io/DOTA/).

## Citing this work

If you found the paper or this codebase useful, please consider citing:

```bibtex
@article{poli2022self,
    title={Self-Similarity Priors: Neural Collages as Differentiable Fractal Representations},
    author={Poli, Michael and Xu, Winnie and Massaroli, Stefano and Meng, Chenlin and Kim, Kuno and Ermon, Stefano}, 
    journal={arXiv preprint arXiv:2204.07673}, 
    year={2022}
      }
```
