# Spherical residual vector quantization (SRVQ)

This repository contains a Pytorch-based minimalist implementation of the spherical residual vector quantization (SRVQ) module used in our Gull neural audio codec framework.

SRVQ is a modification to the standard RVQ to better quantize unit-norm inputs. The general idea is to use unit-norm codebooks with standard VQ-VAE selection and update scheme at the first hierarchy (R=1), while use **rotation matrices** defined by Householder transform as learnable codebooks for other hierarchies (R>1).

# Reference
If you use SRVQ in your project, please consider citing the following paper:

> @article{luo2024gull,  
>  title={Gull: A Generative Multifunctional Audio Codec},  
>  author={Luo, Yi and Yu, Jianwei and Chen, Hangting and Gu Rongzhi and Weng, Chao},  
>  journal={arXiv preprint arXiv:2404.04947},  
>  year={2024}  
> }