# Conditional Diffusion with Ordinal Regression: Longitudinal Data Generation for Neurodegenerative Disease Studies
Official code repository of the paper Conditional Diffusion with Ordinal Regression: Longitudinal Data Generation for Neurodegenerative Disease Studies (ICLR 2025 Spotlight).

## Contribution 
1) We propose a novel generative model for longitudinal data, conditioned on ordinal factors such as age and disease severity. By incorporating ordinal regression into a diffusion model, our method effectively captures the ordinality of conditions.
2) To handle temporal heterogeneity within visits, our method sequentially generates samples that seamlessly fill gaps within sparse data points. These interim samples capture both population-level trends and individual-specific features by a dual-sampling approach, which leads to generating personalized longitudinal data.
3) To maximize the utility of limited data, the framework is extended to enable the model to learn data from different sources with a domain condition. By integrating both time-invariant and time-dependent factors, ConDOR improves generalizability across diverse datasets while capturing common progressive features.

## Datasets
All data required are available through [ADNI](https://adni.loni.usc.edu/) and [OASIS](https://sites.wustl.edu/oasisbrains/).
To obtain cortical thickness, T1-weighted MR images were parcellated into 148 brain regions based on the Destrieux atlas on both datasets, and skull stripping, tissue segmentation, and image registration were performed using Freesurfer.
Other measures such as Amyloid, FDG, and Tau SUVR were calculated from PET scans based on the Destrieux atlas, and the cerebellum was used as the reference region to calculate the SUVR for each modality.

## Pretrained models
ConDOR is comprised of two diffusion models: RDM and TDM. These models are trained separately by using separate loss functions $L_\text{RDM}$ and $L_\text{TDM}$, and we provide both of the pretrained RDM and TDM for all experiments (CT, Amyloid, FDG, and Tau) in [this drive](https://drive.google.com/file/d/19RlXMt4QY05MRQeXyO76wO8qpZURO8WJ/view?usp=sharing).

## Running Experiments
Both RDM and TDM were implemented for single GPU running. 

RDM is trained first followed by TDM. To run RDM from scratch, use the following command:
```
python main.py --warmup=1
```
To train TDM, set the path of the trained RDM in train() of trainer.py.

```
pth = os.path.join('warmup_RDM.pt')
```
Finally, run the following command to train TDM and evaluate the performance of ConDOR (RDM + TDM):

```
python main.py 
```

## Citation
If you would like to cite our paper, please use the BibTeX below.

```
@inproceedings{choconditional,
  title={Conditional Diffusion with Ordinal Regression: Longitudinal Data Generation for Neurodegenerative Disease Studies},
  author={Cho, Hyuna and Wei, Ziquan and Lee, Seungjoo and Dan, Tingting and Wu, Guorong and Kim, Won Hwa},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```




