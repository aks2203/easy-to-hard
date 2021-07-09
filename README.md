# Easy-To-Hard

The official repository for the paper ["Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks"](https://arxiv.org/abs/2106.04537).

## Getting Started
### Requirements    
To install requirements:

```pip install -r requirements.txt```

To use the datasets we use in this project, we recommend you install our [Python package](https://github.com/aks2203/easy-to-hard-data) `easy-to-hard-data` by running:
```
pip install easy-to-hard-data
```

You many also download raw datasets. See the [Google Drive folder](https://drive.google.com/drive/folders/1ad_ZESAddlfx-b3CnK1ohoKz6Sp8U-5g?usp=sharing).

## Training \& Testing
See the dataset specific documentation in the corresonding directories: [Prefix Sums](./prefix_sums/README_PREFIXSUMS.md), [Mazes](./mazes/README_MAZES.md), [Chess](./chess/README_CHESS.md).

## Citing our paper
If you find this code helpful, please consider citing our work.
```
@misc{schwarzschild2021learn,
      title={Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks}, 
      author={Avi Schwarzschild and Eitan Borgnia and Arjun Gupta and Furong Huang and Uzi Vishkin and Micah Goldblum and Tom Goldstein},
      year={2021},
      eprint={2106.04537},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
