# Demo Paper: Self-Augmented Noisy Image for Blind Denoising (sample seed)

> 这是仓库自带的**示例文档**，用来在没有论文语料时快速验证 RAG 链路：
>
> ```bash
> researchops ingest examples/sample_document.md
> researchops search "denoising method"
> ```
>
> 内容为虚构的演示数据，不代表任何真实论文结果。

## Abstract

We propose a self-augmented noisy-image network for blind image denoising. The core
idea is that a network can learn to denoise without clean ground truth by first
generating a synthetic noisy image, then training a second sub-network to map it back
to the input. This sidesteps the paired-clean-data requirement that constrains most
supervised denoisers.

## Method

The method consists of two sub-networks trained end to end. A noisy-image generation
network takes a clean image and an estimated noise level and produces a synthetic noisy
image. A denoising network then reconstructs the clean image from that synthetic noise.
Because both sub-networks share the same encoder backbone, the model stays compact while
still removing spatially varying noise.

## Results

On the CBSD68 dataset at sigma=25, the method achieves PSNR 31.79 dB and SSIM 0.90,
improving on the previous baseline by 0.12 dB. On the noisy Kodak24 set the gain is
larger, with PSNR improving from 30.02 dB to 30.31 dB at sigma=50.

## Discussion

The main limitation is that the noisy-image generation network can collapse to a
near-identity mapping when the input noise is very low. We mitigate this with a
regularization term that keeps the generated noise spatially structured. Future work
extends the same self-supervision idea to raw sensor data.
