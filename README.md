# Weisfeiler and Leman Return with Graph Transformations

Code repository for our paper ['Weisfeiler and Leman Return with Graph Transformations'](https://openreview.net/pdf?id=Oq5mzL-3SUV) accepted at the 'International Workshop on Mining and Learning with Graphs' at ECMLPKDD 2022.

## Installation
We recommend using conda, as that will allows for an easy installation of graph tool. The following will setup your environment to run all models except CWN. To run CWN, follow the setup described in the `cwn` folder.

Create environment:
```shell
conda create --name envname
conda activate envname
```

Install dependencies:
```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
conda install -c conda-forge graph-tool
pip install -r requirements.txt
```

Add the directories to your pythonpath. Let `@PATH` be the path to where this repository is stored.
```shell
export PYTHONPATH=$PYTHONPATH:@PATH:@PATH/cwn/
```

## Run the experiments
We have prepared scripts to rerun the experiments from the papers. Ensure that your shell is in the main folder of this repository.
- MLP `bash Scripts/mlp.sh`
- GIN `bash Scripts/gin.sh`
- CIN (for this you need the setup described in the `cwn` folder!) `bash Scripts/cin.sh`
- GIN+CRE `bash Scripts/gin_cre.sh`
- ESAN `bash Scripts/esan.sh`
- GIN+SBE `bash Scripts/gin_sbe.sh`

After training, the results can be found in the `Results` directory.

## How to Cite
If you make use of our code or ideas please cite

```
@inproceedings{jogl2022WLReturn,
  title={Weisfeiler and Leman Return with Graph Transformations},
  author={Jogl, Fabian and Thiessen, Maximilian and G{\"a}rtner, Thomas},
  booktitle={ECMLPKDD 2022 International Workshop on Mining and Learning with Graphs},
  year={2022}
}
```

## Credits
The code in this repository is based on

```
@inproceedings{bevilacqua2022equivariant,
title={Equivariant Subgraph Aggregation Networks},
author={Beatrice Bevilacqua and Fabrizio Frasca and Derek Lim and Balasubramaniam Srinivasan and Chen Cai and Gopinath Balamurugan and Michael M. Bronstein and Haggai Maron},
booktitle={International Conference on Learning Representations},
year={2022},
}
```

```
@inproceedings{pmlr-v139-bodnar21a,
  title = 	 {Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks},
  author =       {Bodnar, Cristian and Frasca, Fabrizio and Wang, Yuguang and Otter, Nina and Montufar, Guido F and Li{\'o}, Pietro and Bronstein, Michael},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {1026--1037},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
}
```

```
@inproceedings{neurips-bodnar2021b,
  title={Weisfeiler and Lehman Go Cellular: CW Networks},
  author={Bodnar, Cristian and Frasca, Fabrizio and Otter, Nina and Wang, Yu Guang and Li{\`o}, Pietro and Mont{\'u}far, Guido and Bronstein, Michael},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {34},
  year={2021}
}
```

```
@inproceedings{
xu2018how,
title={How Powerful are Graph Neural Networks?},
author={Keyulu Xu and Weihua Hu and Jure Leskovec and Stefanie Jegelka},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=ryGs6iA5Km},
}
```

```
@article{hu2020ogb,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  journal={arXiv preprint arXiv:2005.00687},
  year={2020}
}
```
