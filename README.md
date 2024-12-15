<h1 align="center">Next Token Prediction Towards Multimodal Intelligence</h1>



<p align="center">
<img src="./figs/development.png" width=100%>
<p>




Building on the foundations of language modeling in natural language processing, Next Token Prediction (NTP) has evolved into a versatile training objective for machine learning tasks across various modalities, achieving considerable success in both understanding and generation tasks. This repo features a comprehensive paper and repos collection for the survey: "Next Token Prediction Towards Multimodal Intelligence: A Comprehensive Survey". 


## 🔥🔥 News
- 2024.12.15: We release the survey and this repo at GitHub! Feel free to make pull requests to add the latest work ~ 

## 📑 Table of Contents

<p align="center">
<img src="./figs/pipelines.png" width=100%>
<p>

1. [Awesome Multimodal Tokenizers](#awesome-multimodal-tokenizers)
   - 1.1 [Vision](#vision-tokenizer)
   - 1.2 [Audio](#audio-tokenizer)
2. [Awesome MMNTP Models](#awesome-mmntp-models)
   - 2.1 [Vision](#vision-model)
   - 2.2 [Audio](#audio-model)
3. [Awesome Multimodal Prompt Engineering](#awesome-multimodal-prompt-engineering) 
   - 3.1 [Multimodal ICL](#multimodal-icl)
   - 3.2 [Multimodal CoT](#multimodal-cot)

---







## Awesome Multimodal Tokenizers


<p align="center">
<img src="./figs/tokenizer.png" width=100%>
<p>


### Vision Tokenizer


| **Paper** | **Time** | **Modality** | **Tokenization Type** | **GitHub** |
|-----------|----------|--------------|-----------------------|-------------|
| [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution (QwenVL2-ViT)](https://arxiv.org/abs/2409.12191) | 2024 | Image,Video | Continuous | [![Star](https://img.shields.io/github/stars/QwenLM/Qwen2-VL.svg?style=social&label=Star)](https://github.com/QwenLM/Qwen2-VL) |
| [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905) | 2024 | Image | Discrete | [![Star](https://img.shields.io/github/stars/FoundationVision/VAR.svg?style=social&label=Star)](https://github.com/FoundationVision/VAR) |
| [SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs](https://arxiv.org/abs/2306.17842) | 2023 | Image | Discrete | - |
| [Unified Language-Vision Pretraining in LLM with Dynamic Discrete Visual Tokenization](https://arxiv.org/abs/2309.04669) | 2023 | Image | Discrete | [![Star](https://img.shields.io/github/stars/jy0205/LaVIT.svg?style=social&label=Star)](https://github.com/jy0205/LaVIT) |
| [Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation](https://arxiv.org/abs/2310.05737) | 2023 | Image,Video | Discrete | - |
| [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](https://arxiv.org/abs/2312.14238) | 2023 | Image | Continuous | [![Star](https://img.shields.io/github/stars/OpenGVLab/InternVL.svg?style=social&label=Star)](https://github.com/OpenGVLab/InternVL) |
| [Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304) | 2023 | Image | Continuous | - |
| [Planting a SEED of Vision in Large Language Model](https://arxiv.org/abs/2307.08041) | 2023 | Image | Discrete | [![Star](https://img.shields.io/github/stars/AILab-CVC/SEED.svg?style=social&label=Star)](https://github.com/AILab-CVC/SEED) |
| [SAM-CLIP: Merging Vision Foundation Models towards Semantic and Spatial Understanding](https://arxiv.org/abs/2310.15308) | 2023 | Image | Continuous | - |
| [EVA-CLIP: Improved Training Techniques for CLIP at Scale](https://arxiv.org/abs/2303.15389) | 2023 | Image | Continuous | [Github](https://github.com/baaivision/EVA/tree/master/EVA-CLIP) |
| [MAGVIT: Masked Generative Video Transformer](https://arxiv.org/abs/2212.05199) | 2022 | Video | Discrete | [![Star](https://img.shields.io/github/stars/google-research/magvit.svg?style=social&label=Star)](https://github.com/google-research/magvit) |
| [Phenaki: Variable Length Video Generation From Open Domain Textual Description](https://arxiv.org/abs/2210.02399) | 2022 | Video | Discrete | - |
| [CoCa: Contrastive Captioners are Image-Text Foundation Models](https://arxiv.org/abs/2205.01917) | 2022 | Image | Continuous | - |
| [Autoregressive Image Generation using Residual Quantization](https://arxiv.org/abs/2203.01941) | 2022 | Image | Discrete | - |
| [ACAV100M: Automatic Curation of Large-Scale Datasets for Audio-Visual Video Representation Learning](https://arxiv.org/pdf/2101.10803) | 2022 | Image | Continuous | [![Star](https://img.shields.io/github/stars/sangho-vision/acav100m.svg?style=social&label=Star)](https://github.com/sangho-vision/acav100m) |
| [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/abs/2212.08013) | 2022 | Image | Continuous | [![Star](https://img.shields.io/github/stars/bwconrad/flexivit.svg?style=social&label=Star)](https://github.com/bwconrad/flexivit) |
| [Vector-quantized Image Modeling with Improved VQGAN](https://arxiv.org/pdf/2110.04627) | 2021 | Image | Discrete | - |
| [ViViT: A Video Vision Transformer](https://arxiv.org/pdf/2103.15691) | 2021 | Video | Continuous | [Github](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit) |
| [BEIT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254) | 2021 | Image | Continuous | [Github](https://github.com/microsoft/unilm/tree/master/beit) |
| [High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/abs/2102.06171) | 2021 | Image | Continuous | [Github](https://github.com/google-deepmind/deepmind-research/tree/master/nfnets) |
| [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) | 2021 | Image | Continuous | [![Star](https://img.shields.io/github/stars/OpenAI/CLIP.svg?style=social&label=Star)](https://github.com/OpenAI/CLIP) |
| [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841) | 2020 | Image | Discrete | [![Star](https://img.shields.io/github/stars/dome272/VQGAN-pytorch.svg?style=social&label=Star)](https://github.com/dome272/VQGAN-pytorch) |
| [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446) | 2019 | Image | Discrete | [![Star](https://img.shields.io/github/stars/rosinality/vq-vae-2-pytorch.svg?style=social&label=Star)](https://github.com/rosinality/vq-vae-2-pytorch) |
| [Temporal 3D ConvNets: New Architecture and Transfer Learning for Video Classification](https://arxiv.org/abs/1711.08200) | 2017 | Video | Continuous | [![Star](https://img.shields.io/github/stars/MohsenFayyaz89/T3D.svg?style=social&label=Star)](https://github.com/MohsenFayyaz89/T3D) |
| [Neural Discrete Representation Learning (VQVAE)](https://arxiv.org/abs/1711.00937) | 2017 | Image, Video, Audio | Discrete | - |


### Audio Tokenizer
| **Paper** | **Time** | **Modality** | **Tokenization Type** | **GitHub** |
|-----------|----------|--------------|----------|----------|
| [Moshi: a speech-text foundation model for real-time dialogue (Mimi)](https://arxiv.org/abs/2410.00037) | 2024  | Audio | Discrete | [![Star](https://img.shields.io/github/stars/kyutai-labs/moshi.svg?style=social&label=Star)](https://github.com/kyutai-labs/moshi) |
| [WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling](https://arxiv.org/abs/2408.16532) | 2024  | Audio | Discrete | [![Star](https://img.shields.io/github/stars/jishengpeng/WavTokenizer.svg?style=social&label=Star)](https://github.com/jishengpeng/WavTokenizer) |
| [SemantiCodec: An Ultra Low Bitrate Semantic Audio Codec for General Sound](https://arxiv.org/abs/2405.00233) | 2024  | Audio | Discrete | [![Star](https://img.shields.io/github/stars/haoheliu/SemantiCodec-inference.svg?style=social&label=Star)](https://github.com/haoheliu/SemantiCodec-inference) |
| [NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models (FACodec)](https://arxiv.org/abs/2403.03100) | 2024  | Audio | Discrete | - |
| [SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models](https://arxiv.org/abs/2308.16692) | 2023  | Audio | Discrete | [![Star](https://img.shields.io/github/stars/ZhangXInFD/SpeechTokenizer.svg?style=social&label=Star)](https://github.com/ZhangXInFD/SpeechTokenizer) |
| [HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec](https://arxiv.org/abs/2305.02765) | 2023  | Audio | Discrete | [![Star](https://img.shields.io/github/stars/yangdongchao/AcademiCodec.svg?style=social&label=Star)](https://github.com/yangdongchao/AcademiCodec) |
| [LMCodec: A Low Bitrate Speech Codec With Causal Transformer Models](https://arxiv.org/abs/2303.12984) | 2023  | Audio | Discrete | - |
| [High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) | 2023  | Audio | Discrete | [![Star](https://img.shields.io/github/stars/descriptinc/descript-audio-codec.svg?style=social&label=Star)](https://github.com/descriptinc/descript-audio-codec) |
| [Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages](https://arxiv.org/abs/2303.01037) | 2023  | Audio | Continuous | - |
| [High Fidelity Neural Audio Compression (Encodec)](https://arxiv.org/abs/2210.13438) | 2022  | Audio | Discrete | [![Star](https://img.shields.io/github/stars/facebookresearch/encodec.svg?style=social&label=Star)](https://github.com/facebookresearch/encodec) |
| [CLAP: Learning Audio Concepts From Natural Language Supervision](https://arxiv.org/abs/2206.04769) | 2022  | Audio | Continuous | [![Star](https://img.shields.io/github/stars/LAION-AI/CLAP.svg?style=social&label=Star)](https://github.com/LAION-AI/CLAP) |
| [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356) | 2022  | Audio | Continuous | [![Star](https://img.shields.io/github/stars/openai/whisper.svg?style=social&label=Star)](https://github.com/openai/whisper) |
| [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/abs/2202.03555) | 2022  | Audio | Continuous | [![Star](https://img.shields.io/github/stars/facebookresearch/fairseq.svg?style=social&label=Star)](https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/README.md) |
| [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) | 2021   | Audio | Continuous | [![Star](https://img.shields.io/github/stars/microsoft/unilm.svg?style=social&label=Star)](https://github.com/microsoft/unilm/tree/master/wavlm) |
| [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) | 2021   | Audio | Continuous | [![Star](https://img.shields.io/github/stars/facebookresearch/fairseq.svg?style=social&label=Star)](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md) |
| [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312) | 2021     | Audio | Discrete | - |
| [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) | 2020     | Audio | Continuous | [![Star](https://img.shields.io/github/stars/facebookresearch/fairseq.svg?style=social&label=Star)](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md) |
| [vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations](https://arxiv.org/abs/1910.05453) | 2019     | Audio | Discrete | [![Star](https://img.shields.io/github/stars/facebookresearch/fairseq.svg?style=social&label=Star)](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md) |


## Awesome MMNTP Models

<p align="center">
<img src="./figs/model_type.png" width=100%>
<p>

### Vision Model

| **Paper** | **Time** | **Modality** | **Model Type** | **Task** | **GitHub** |
|-----------|----------|--------------|----------------|----------|------------|
| [Randomized Autoregressive Visual Generation (RAR)](https://arxiv.org/abs/2411.00776) | 2024 | Image | Unified | Text2Image | [![Star](https://img.shields.io/github/stars/bytedance/1d-tokenizer.svg?style=social&label=Star)](https://github.com/bytedance/1d-tokenizer) |
| [Mono-InternVL: Pushing the Boundaries of Monolithic Multimodal Large Language Models with Endogenous Visual Pre-training (MonoInternVL)](https://arxiv.org/abs/2410.08202) | 2024 | Image | Unified | Image2Text | - |
| [A Single Transformer for Scalable Vision-Language Modeling (SOLO)](https://arxiv.org/abs/2407.06438) | 2024 | Image | Unified | Image2Text | - |
| [Unveiling Encoder-Free Vision-Language Models (EVE)](https://arxiv.org/abs/2406.11832) | 2024 | Image | Unified | Image2Text | [![Star](https://img.shields.io/github/stars/baaivision/EVE.svg?style=social&label=Star)](https://github.com/baaivision/EVE) |
| [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution (Qwen2VL)](https://arxiv.org/abs/2409.12191) | 2024 | Image | Compositional | Image2Text | [![Star](https://img.shields.io/github/stars/QwenLM/Qwen2-VL.svg?style=social&label=Star)](https://github.com/QwenLM/Qwen2-VL) |
| [Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation (Janus)](https://arxiv.org/abs/2410.13848) | 2024 | Image | Compositional | Image2Text, Text2Image | [![Star](https://img.shields.io/github/stars/deepseek-ai/Janus.svg?style=social&label=Star)](https://github.com/deepseek-ai/Janus) |
| [Emu3: Next-Token Prediction is All You Need (Emu3)](https://arxiv.org/abs/2409.18869) | 2024 | Image, Video | Unified | Image2Text, Text2Image, Text2Video | [![Star](https://img.shields.io/github/stars/baaivision/Emu3.svg?style=social&label=Star)](https://github.com/baaivision/Emu3) |
| [Show-o: One Single Transformer to Unify Multimodal Understanding and Generation (Show-o)](https://arxiv.org/abs/2408.12528) | 2024 | Image, Video | Unified | Image2Text, Text2Image, Text2Video | [![Star](https://img.shields.io/github/stars/showlab/Show-o.svg?style=social&label=Star)](https://github.com/showlab/Show-o) |
| [VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation (VILA-U)](https://arxiv.org/abs/2409.04429) | 2024 | Image, Video | Unified | Image2Text, Text2Image, Text2Video | [![Star](https://img.shields.io/github/stars/mit-han-lab/vila-u.svg?style=social&label=Star)](https://github.com/mit-han-lab/vila-u) |
| [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model (Transfusion)](https://arxiv.org/abs/2408.11039) | 2024 | Image | Unified | Image2Text | - |
| [Fluid: Scaling Autoregressive Text-to-image Generative Models with Continuous Tokens (Fluid)](https://arxiv.org/abs/2410.13863) | 2024 | Image | Unified | Image2Text | - |
| [Autoregressive Image Generation without Vector Quantization (MAR)](https://arxiv.org/abs/2406.11838) | 2024 | Image | Unified | Image2Text | [![Star](https://img.shields.io/github/stars/LTH14/mar.svg?style=social&label=Star)](https://github.com/LTH14/mar) |
| [Chameleon: Mixed-Modal Early-Fusion Foundation Models (Chameleon)](https://arxiv.org/abs/2405.09818) | 2024 | Image | Unified | Image2Text, Text2Image | [![Star](https://img.shields.io/github/stars/facebookresearch/chameleon.svg?style=social&label=Star)](https://github.com/facebookresearch/chameleon) |
| [Mini-Gemini: Mining the Potential of Multi-modality Vision Language Models (Mini-Genimi)](https://arxiv.org/abs/2403.18814) | 2024 | Image | Compositional | Image2Text, Text2Image | [![Star](https://img.shields.io/github/stars/dvlab-research/MGM.svg?style=social&label=Star)](https://github.com/dvlab-research/MGM) |
| [A Spark of Vision-Language Intelligence: 2-Dimensional Autoregressive Transformer for Efficient Finegrained Image Generation (DnD-Transformer)](https://arxiv.org/abs/2410.01912) | 2024 | Image | Unified | Text2Image | [![Star](https://img.shields.io/github/stars/chenllliang/DnD-Transformer.svg?style=social&label=Star)](https://github.com/chenllliang/DnD-Transformer) |
| [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction (VAR)](https://arxiv.org/abs/2404.02905) | 2024 | Image | Unified | Text2Image | [![Star](https://img.shields.io/github/stars/FoundationVision/VAR.svg?style=social&label=Star)](https://github.com/FoundationVision/VAR) |
| [Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation (LlamaGen)](https://arxiv.org/abs/2406.06525) | 2024 | Image | Unified | Text2Image | [![Star](https://img.shields.io/github/stars/FoundationVision/LlamaGen.svg?style=social&label=Star)](https://github.com/FoundationVision/LlamaGen) |
| [MiniGPT-5: Interleaved Vision-and-Language Generation via Generative Vokens (MiniGPT5)](https://arxiv.org/abs/2310.02239) | 2023 | Image | Compositional | Image2Text, Text2Image | [![Star](https://img.shields.io/github/stars/eric-ai-lab/MiniGPT-5.svg?style=social&label=Star)](https://github.com/eric-ai-lab/MiniGPT-5) |
| [BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing (Blip-Diffusion)](https://arxiv.org/abs/2305.14720) | 2023 | Image | Compositional | Text2Image | [![Star](https://img.shields.io/github/stars/salesforce/LAVIS.svg?style=social&label=Star)](https://github.com/salesforce/LAVIS) |
| [Kosmos-G: Generating Images in Context with Multimodal Large Language Models (Kosmos-G)](https://arxiv.org/abs/2310.02992) | 2023 | Image | Compositional | Text2Image | [![Star](https://img.shields.io/github/stars/xichenpan/Kosmos-G.svg?style=social&label=Star)](https://github.com/xichenpan/Kosmos-G) |
| [Unified Language-Vision Pretraining in LLM with Dynamic Discrete Visual Tokenization (LaVIT)](https://arxiv.org/abs/2309.04669) | 2023 | Image | Compositional | Image2Text, Text2Image | [![Star](https://img.shields.io/github/stars/jy0205/LaVIT.svg?style=social&label=Star)](https://github.com/jy0205/LaVIT) |
| [Generative Multimodal Models are In-Context Learners (Emu2)](https://arxiv.org/abs/2312.13286) | 2023 | Image | Compositional | Image2Text, Text2Image | [![Star](https://img.shields.io/github/stars/baaivision/Emu.svg?style=social&label=Star)](https://github.com/baaivision/Emu) |
| [Generative Pretraining in Multimodality (Emu1)](https://arxiv.org/abs/2307.05222) | 2023 | Image | Compositional | Image2Text, Text2Image | [![Star](https://img.shields.io/github/stars/baaivision/Emu.svg?style=social&label=Star)](https://github.com/baaivision/Emu) |
| [Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action (Unified-IO2)](https://arxiv.org/abs/2312.17172) | 2023 | Image, Video, Audio | Compositional | Image2Text, Text2Image, Audio2Text, Text2Audio, Text2Video | [![Star](https://img.shields.io/github/stars/allenai/unified-io-2.svg?style=social&label=Star)](https://github.com/allenai/unified-io-2) |
| [Language Is Not All You Need: Aligning Perception with Language Models (Kosmos-1)](https://arxiv.org/abs/2302.14045) | 2023 | Image | Compositional | Image2Text | [![Star](https://img.shields.io/github/stars/microsoft/unilm.svg?style=social&label=Star)](https://github.com/microsoft/unilm) |
| [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks (InternVL)](https://arxiv.org/abs/2312.14238) | 2023 | Image | Compositional | Image2Text | [![Star](https://img.shields.io/github/stars/OpenGVLab/InternVL.svg?style=social&label=Star)](https://github.com/OpenGVLab/InternVL) |
| [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond (QwenVL)](https://arxiv.org/abs/2308.12966) | 2023 | Image | Compositional | Image2Text | [![Star](https://img.shields.io/github/stars/QwenLM/Qwen-VL.svg?style=social&label=Star)](https://github.com/QwenLM/Qwen-VL) |
| [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models (Molom)](https://arxiv.org/abs/2409.17146) | 2023 | Image | Compositional | Image2Text | -) |
| [Fuyu-8B: A Multimodal Architecture for AI Agents (Fuyu)](https://www.adept.ai/blog/fuyu-8b) | 2023 | Image | Unified | Image2Text | - |
| [Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models (BLIP2)](https://arxiv.org/abs/2301.12597) | 2023 | Image | Compositional | Image2Text | [![Star](https://img.shields.io/github/stars/salesforce/LAVIS.svg?style=social&label=Star)](https://github.com/salesforce/LAVIS) |
| [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) | 2023 | Image | Compositional | Image2Text | [![Star](https://img.shields.io/github/stars/haotian-liu/LLaVA.svg?style=social&label=Star)](https://github.com/haotian-liu/LLaVA) |
| [MiniGPT4: a Visual Language Model for Few-Shot Learning (MiniGPT4)](https://arxiv.org/abs/2204.14198) | 2022 | Image | Compositional | Image2Text | - |
| [Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks (Unified-IO)](https://arxiv.org/abs/2206.08916) | 2022 | Image | Compositional | Image2Text, Text2Image | - |
| [Zero-Shot Text-to-Image Generation (DALLE)](https://arxiv.org/abs/2102.12092) | 2022 | Image | Unified | Text2Image | - |
| [Flamingo: a Visual Language Model for Few-Shot Learning (Flamingo)](https://arxiv.org/abs/2204.14198) | 2022 | Image | Compositional | Image2Text | - |

### Audio Model

| **Paper** | **Time** | **Modality** | **Model Type** | **Task** | **GitHub** |
|-----------|----------|--------------|----------------|----------|------------|
| [VoxtLM: Unified Decoder-Only Models for Consolidating Speech Recognition, Synthesis and Speech, Text Continuation Tasks (VoxtLM)](https://arxiv.org/abs/2309.07937) | 2024 | Audio | Unified | A2T, T2A, A2A, T2T | - |
| [Moshi: a speech-text foundation model for real-time dialogue (Moshi)](https://arxiv.org/abs/2410.00037) | 2024 | Audio | Unified | A2A | [![Star](https://img.shields.io/github/stars/kyutai-labs/moshi.svg?style=social&label=Star)](https://github.com/kyutai-labs/moshi) |
| [Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming (Mini-Omni)](https://arxiv.org/abs/2408.16725) | 2024 | Audio | Compositional | A2A | [![Star](https://img.shields.io/github/stars/gpt-omni/mini-omni.svg?style=social&label=Star)](https://github.com/gpt-omni/mini-omni) |
| [LLaMA-Omni: Seamless Speech Interaction with Large Language Models (LLaMA-Omni)](https://arxiv.org/abs/2409.06666) | 2024 | Audio | Compositional | A2A | [![Star](https://img.shields.io/github/stars/ictnlp/LLaMA-Omni.svg?style=social&label=Star)](https://github.com/ictnlp/LLaMA-Omni) |
| [SpeechVerse: A Large-scale Generalizable Audio Language Model (SpeechVerse)](https://arxiv.org/abs/2405.08295) | 2024 | Audio | Compositional | A2T | - |
| [Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities (AudioFlamingo)](https://arxiv.org/abs/2402.01831) | 2024 | Audio | Compositional | A2T | [![Star](https://img.shields.io/github/stars/NVIDIA/audio-flamingo.svg?style=social&label=Star)](https://github.com/NVIDIA/audio-flamingo) |
| [WavLLM: Towards Robust and Adaptive Speech Large Language Model (WavLLM)](https://arxiv.org/abs/2404.00656) | 2024 | Audio | Compositional | A2T | [![Star](https://img.shields.io/github/stars/microsoft/SpeechT5.svg?style=social&label=Star)](https://github.com/microsoft/SpeechT5) |
| [MELLE: Autoregressive Speech Synthesis without Vector Quantization](https://arxiv.org/abs/2407.08551) | 2024 | Audio | Unified | T2A | - |
| [Seed-TTS: A Family of High-Quality Versatile Speech Generation Models (Seed-TTS)](https://arxiv.org/abs/2406.02430) | 2024 | Audio | Compositional | T2A | - |
| [FireRedTTS: A Foundation Text-To-Speech Framework for Industry-Level Generative Speech Applications (FireRedTTS)](https://www.arxiv.org/abs/2409.03283) | 2024 | Audio | Compositional | T2A | [![Star](https://img.shields.io/github/stars/FireRedTeam/FireRedTTS.svg?style=social&label=Star)](https://github.com/FireRedTeam/FireRedTTS) |
| [CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens (CosyVoice)](https://arxiv.org/abs/2407.05407) | 2024 | Audio | Compositional | T2A | [![Star](https://img.shields.io/github/stars/FunAudioLLM/CosyVoice.svg?style=social&label=Star)](https://github.com/FunAudioLLM/CosyVoice) |
| [Uniaudio: An audio foundation model toward universal audio generation (UniAudio)](https://arxiv.org/abs/2310.00704) | 2024 | Audio | Unified | T2A, A2A | [![Star](https://img.shields.io/github/stars/yangdongchao/UniAudio.svg?style=social&label=Star)](https://github.com/yangdongchao/UniAudio) |
| [BASE TTS: Lessons from building a billion-parameter text-to-speech model on 100K hours of data (BASE TTS)](https://arxiv.org/abs/2402.08093) | 2024 | Audio | Unified | T2A | - |
| [VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild (VoiceCraft)](https://arxiv.org/abs/2403.16973) | 2024 | Audio | Unified | T2A | [![Star](https://img.shields.io/github/stars/jasonppy/VoiceCraft.svg?style=social&label=Star)](https://github.com/jasonppy/VoiceCraft) |
| [Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities (SpeechGPT)](https://arxiv.org/abs/2305.11000) | 2023 | Audio | Unified | A2T, T2A, A2A, T2T | [![Star](https://img.shields.io/github/stars/0nutation/SpeechGPT.svg?style=social&label=Star)](https://github.com/0nutation/SpeechGPT) |
| [Lauragpt: Listen, attend, understand, and regenerate audio with gpt (LauraGPT)](https://arxiv.org/abs/2310.04673) | 2023 | Audio | Unified | A2T, T2A, A2A, T2T | - |
| [Viola: Unified codec language models for speech recognition, synthesis, and translation (VIOLA)](https://arxiv.org/abs/2305.16107) | 2023 | Audio | Compositional | A2T, T2A, A2A, T2T | - |
| [Audiopalm: A large language model that can speak and listen (AudioPaLM)](https://arxiv.org/abs/2306.12925) | 2023 | Audio | Compositional | A2T, T2A, A2A | - |
| [Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models (Qwen-Audio)](https://arxiv.org/abs/2311.07919) | 2023 | Audio | Compositional | A2T | [![Star](https://img.shields.io/github/stars/QwenLM/Qwen-Audio.svg?style=social&label=Star)](https://github.com/QwenLM/Qwen-Audio) |
| [Salmonn: Towards generic hearing abilities for large language models (SALMONN)](https://arxiv.org/abs/2310.13289) | 2023 | Audio | Compositional | A2T | [![Star](https://img.shields.io/github/stars/bytedance/SALMONN.svg?style=social&label=Star)](https://github.com/bytedance/SALMONN) |
| [On decoder-only architecture for speech-to-text and large language model integration (SpeechLLaMA)](https://arxiv.org/abs/2307.03917) | 2023 | Audio | Compositional | A2T | - |
| [Listen, think, and understand (LTU)](https://arxiv.org/abs/2305.10790) | 2023 | Audio | Compositional | A2T | [![Star](https://img.shields.io/github/stars/YuanGongND/ltu.svg?style=social&label=Star)](https://github.com/YuanGongND/ltu) |
| [Pengi: An audio language model for audio tasks (Pengi)](https://arxiv.org/abs/2305.11834) | 2023 | Audio | Compositional | A2T | [![Star](https://img.shields.io/github/stars/microsoft/Pengi.svg?style=social&label=Star)](https://github.com/microsoft/Pengi) |
| [Music Understanding LLaMA: Advancing Text-to-Music Generation with Question Answering and Captioning (MU-LLaMA)](https://arxiv.org/abs/2308.11276) | 2023 | Audio | Compositional | A2T | - |
| [SpeechGen: Unlocking the Generative Power of Speech Language Models with Prompts (SpeechGen)](https://arxiv.org/abs/2306.02207) | 2023 | Audio | Unified | T2A | [![Star](https://img.shields.io/github/stars/ga642381/SpeechGen.svg?style=social&label=Star)](https://github.com/ga642381/SpeechGen) |
| [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) | 2023 | Audio | Compositional | T2A | [![Star](https://img.shields.io/github/stars/microsoft/unilm.svg?style=social&label=Star)](https://github.com/microsoft/unilm) |
| [Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) | 2023 | Audio | Unified | T2A | [![Star](https://img.shields.io/github/stars/facebookresearch/audiocraft.svg?style=social&label=Star)](https://github.com/facebookresearch/audiocraft) |
| [Make-A-Voice: Unified Voice Synthesis With Discrete Representation (Make-A-Voice)](https://arxiv.org/abs/2305.19269) | 2023 | Audio | Compositional | T2A | - |
| [Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision (SPEAR-TTS)](https://arxiv.org/abs/2302.03540) | 2023 | Audio | Compositional | T2A | - |
| [AudioGen: Textually Guided Audio Generation (AudioGen)](https://arxiv.org/abs/2209.15352) | 2022 | Audio | Unified | T2A | - |
| [AudioLM: a Language Modeling Approach to Audio Generation (AudioLM)](https://arxiv.org/abs/2209.03143) | 2022 | Audio | Compositional | A2A | - |
| [Generative Spoken Language Modeling from Raw Audio (GSLM)](https://arxiv.org/abs/2102.01192) | 2021 | Audio | Unified | A2A | - |


## Awesome Multimodal Prompt Engineering 

<p align="center">
<img src="./figs/prompt.png" width=100%>
<p>

### Multimodal ICL

| **Paper** | **Time** | **Modality** | **GitHub** |
|-----------|----------|--------------|-------------|
| [Multimodal Few-Shot Learning with Frozen Language Models (Frozen)](https://arxiv.org/abs/2106.13884) | 2021 | Image | - |
| [Flamingo: a Visual Language Model for Few-Shot Learning (Flamingo)](https://arxiv.org/abs/2204.14198) | 2022 | Image | - |
| [MMICL: Empowering Vision-language Model with Multi-Modal In-Context Learning (MMICL)](https://arxiv.org/abs/2309.07915) | 2023 | Image | [![Star](https://img.shields.io/github/stars/HaozheZhao/MIC.svg?style=social&label=Star)](https://github.com/HaozheZhao/MIC) |
| [Efficient In-Context Learning in Vision-Language Models for Egocentric Videos (EILeV)](https://arxiv.org/abs/2311.17041) | 2023 | Image | [![Star](https://img.shields.io/github/stars/yukw777/EILEV.svg?style=social&label=Star)](https://yukw777.github.io/EILEV/) |
| [OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models (Open-Flamingo)](https://arxiv.org/abs/2308.01390) | 2023 | Image | [![Star](https://img.shields.io/github/stars/mlfoundations/open_flamingo.svg?style=social&label=Star)](https://github.com/mlfoundations/open_flamingo) |
| [Link-Context Learning for Multimodal LLMs (LCL)](https://arxiv.org/abs/2308.07891) | 2023 | Image | [![Star](https://img.shields.io/github/stars/isekai-portal/Link-Context-Learning.svg?style=social&label=Star)](https://github.com/isekai-portal/Link-Context-Learning) |
| [Med-Flamingo: a Multimodal Medical Few-shot Learner (Med-Flamingo)](https://arxiv.org/abs/2307.15189) | 2023 | Image | [![Star](https://img.shields.io/github/stars/snap-stanford/med-flamingo.svg?style=social&label=Star)](https://github.com/snap-stanford/med-flamingo) |
| [MIMIC-IT: Multi-Modal In-Context Instruction Tuning (MIMIC-IT)](https://arxiv.org/abs/2306.05425) | 2023 | Image | [![Star](https://img.shields.io/github/stars/Luodian/otter.svg?style=social&label=Star)](https://github.com/Luodian/otter) |
| [Sequential Modeling Enables Scalable Learning for Large Vision Models (LVM)](https://arxiv.org/abs/2312.00785) | 2023 | Image | [![Star](https://img.shields.io/github/stars/ytongbai/LVM.svg?style=social&label=Star)](https://github.com/ytongbai/LVM) |
| [World Model on Million-Length Video And Language With Blockwise RingAttention (LWM)](https://arxiv.org/abs/2402.08268) | 2023 | Image, Video | [![Star](https://img.shields.io/github/stars/largeworldmodel/lwm.svg?style=social&label=Star)](https://largeworldmodel.github.io/lwm/) |
| [Exploring Diverse In-Context Configurations for Image Captioning (Yang et al.)](https://github.com/yongliang-wu/ExploreCfg) | 2024 | Image | [![Star](https://img.shields.io/github/stars/yongliang-wu/ExploreCfg.svg?style=social&label=Star)](https://github.com/yongliang-wu/ExploreCfg) |
| [Visual In-Context Learning for Large Vision-Language Models (VisualICL)](https://aclanthology.org/2024.findings-acl.940/) | 2024 | Image | - |
| [Many-Shot In-Context Learning in Multimodal Foundation Models (Many-Shots ICL)](https://arxiv.org/abs/2405.09798) | 2024 | Image | [![Star](https://img.shields.io/github/stars/stanfordmlgroup/ManyICL.svg?style=social&label=Star)](https://github.com/stanfordmlgroup/ManyICL) |
| [Can MLLMs Perform Text-to-Image In-Context Learning? (CoBSAT)](https://arxiv.org/abs/2402.01293) | 2024 | Image | [![Star](https://img.shields.io/github/stars/UW-Madison-Lee-Lab/CoBSAT.svg?style=social&label=Star)](https://github.com/UW-Madison-Lee-Lab/CoBSAT) |
| [Video In-context Learning (Video ICL)](https://arxiv.org/abs/2407.07356) | 2024 | Video | [![Star](https://img.shields.io/github/stars/aka-ms/vid-icl.svg?style=social&label=Star)](https://aka.ms/vid-icl) |
| [Generative Pretraining in Multimodality (Emu)](https://arxiv.org/abs/2307.05222) | 2024 | Image, Video | [![Star](https://img.shields.io/github/stars/baaivision/Emu.svg?style=social&label=Star)](https://github.com/baaivision/Emu) |
| [Generative Multimodal Models are In-Context Learners (Emu2)](https://arxiv.org/abs/2312.13286) | 2024 | Image, Video | [![Star](https://img.shields.io/github/stars/baaivision/emu2.svg?style=social&label=Star)](https://baaivision.github.io/emu2) |
| [Towards More Unified In-context Visual Understanding (Sheng et al.)](https://arxiv.org/abs/2312.02520) | 2024 | Image | - |
| [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) | 2023 | Audio | [![Star](https://img.shields.io/github/stars/microsoft/unilm.svg?style=social&label=Star)](https://github.com/microsoft/unilm) |
| [MELLE: Autoregressive Speech Synthesis without Vector Quantization (MELLE)](https://arxiv.org/abs/2407.08551) | 2024 | Audio | - |
| [Seed-TTS: A Family of High-Quality Versatile Speech Generation Models (Seed-TTS)](https://arxiv.org/abs/2406.02430) | 2024 | Audio | - |
| [Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities (Audio Flamingo)](https://arxiv.org/abs/2402.01831) | 2024 | Audio | [![Star](https://img.shields.io/github/stars/NVIDIA/audio-flamingo.svg?style=social&label=Star)](https://github.com/NVIDIA/audio-flamingo) |
| [Moshi: a speech-text foundation model for real-time dialogue (Moshi)](https://arxiv.org/abs/2410.00037) | 2024 | Audio | [![Star](https://img.shields.io/github/stars/kyutai-labs/moshi.svg?style=social&label=Star)](https://github.com/kyutai-labs/moshi) |



### Multimodal CoT

| **Paper** | **Time** | **Modality**  | **GitHub** |
|-----------|----------|--------------|-------------|
| [WavLLM: Towards Robust and Adaptive Speech Large Language Model (WavLLM)](https://arxiv.org/abs/2404.00656) | 2024 | Audio | [![Star](https://img.shields.io/github/stars/microsoft/SpeechT5.svg?style=social&label=Star)](https://github.com/microsoft/SpeechT5) |
| [SpeechVerse: A Large-scale Generalizable Audio Language Model (SpeechVerse)](https://arxiv.org/abs/2405.08295) | 2024 | Audio | - |
| [CoT-ST: Enhancing LLM-based Speech Translation with Multimodal Chain-of-Thought](https://arxiv.org/abs/2409.19510) | 2024 | Audio | [![Star](https://img.shields.io/github/stars/X-LANCE/SLAM-LLM.svg?style=social&label=Star)](https://github.com/X-LANCE/SLAM-LLM/tree/main/examples/st_covost2) |
| [Chain-of-Thought Prompting for Speech Translation](https://arxiv.org/abs/2409.11538) | 2024 | Audio | - |
| [Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition](https://openreview.net/pdf?id=fO31YAyNbI) | 2024 | Video | [![Star](https://img.shields.io/github/stars/scofield7419/Video-of-Thought.svg?style=social&label=Star)](https://github.com/scofield7419/Video-of-Thought)|
| [VideoCoT: A Video Chain-of-Thought Dataset with Active Annotation Tool](https://arxiv.org/pdf/2407.05355) | 2024 | Video | - |
| [Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning](https://arxiv.org/pdf/2403.16999) | 2024 | Image | [![Star](https://img.shields.io/github/stars/deepcs233/Visual-CoT.svg?style=social&label=Star)](https://github.com/deepcs233/Visual-CoT)|
| [CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations](https://arxiv.org/pdf/2402.04236) | 2024 | Image | [![Star](https://img.shields.io/github/stars/THUDM/CogCoM.svg?style=social&label=Star)](https://github.com/THUDM/CogCoM)|
| [Compositional Chain-of-Thought Prompting for Large Multimodal Model](https://arxiv.org/pdf/2311.17076) | 2023 | Image | [![Star](https://img.shields.io/github/stars/chancharikmitra/CCoT?style=social&label=Star)](https://github.com/chancharikmitra/CCoT)|
| [V∗: Guided Visual Search as a Core Mechanism in Multimodal LLMs](https://arxiv.org/pdf/2312.14135) | 2023 | Image | [![Star](https://img.shields.io/github/stars/penghao-wu/vstar.svg?style=social&label=Star)](https://github.com/penghao-wu/vstar)|
| [DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning in Language Models](https://arxiv.org/pdf/2310.16436) | 2023 | Image | [![Star](https://img.shields.io/github/stars/SooLab/DDCOT.svg?style=social&label=Star)](https://github.com/SooLab/DDCOT)|
| [Visual Chain-of-Thought Diffusion Models](https://arxiv.org/pdf/2303.16187) | 2023 | Image | - |
| [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/pdf/2302.00923) | 2023 | Image |[![Star](https://img.shields.io/github/stars/amazon-science/mm-cot.svg?style=social&label=Star)](https://github.com/amazon-science/mm-cot)|







## Reference

coming soon!





