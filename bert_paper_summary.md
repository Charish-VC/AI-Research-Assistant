# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## Abstract

We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing benchmarks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

## Introduction

Language model pre-training has been shown to be effective for improving many natural language processing tasks. These include sentence-level tasks such as natural language inference and paraphrasing, which aim to predict the relationships between sentences by analyzing them holistically, as well as token-level tasks such as named entity recognition and question answering, where models are required to produce fine-grained output at the token level.

There are two existing strategies for applying pre-trained language representations to downstream tasks: feature-based and fine-tuning. The feature-based approach, such as ELMo, uses task-specific architectures that include the pre-trained representations as additional features. The fine-tuning approach, such as the Generative Pre-trained Transformer (OpenAI GPT), introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning all pre-trained parameters.

The two approaches share the same objective function during pre-training, where they use unidirectional language models to learn general language representations. We argue that current techniques restrict the power of the pre-trained representations, especially for the fine-tuning approaches. The major limitation is that standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training.

## The BERT Model Architecture

BERT's model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017). The transformer architecture uses self-attention mechanism to process input sequences in parallel rather than sequentially.

BERT uses the transformer encoder architecture. The input representation is constructed by summing the corresponding token, segment, and position embeddings. The model uses WordPiece embeddings with a 30,000 token vocabulary.

We denote the number of layers (i.e., Transformer blocks) as L, the hidden size as H, and the number of self-attention heads as A. We primarily report results on two model sizes: BERT-BASE (L=12, H=768, A=12, Total Parameters=110M) and BERT-LARGE (L=24, H=1024, A=16, Total Parameters=340M).

## Pre-training Tasks

### Masked Language Model (MLM)

In order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, and then predict those masked tokens. We refer to this procedure as a masked LM (MLM), although it is often referred to as a Cloze task in the literature. In this case, the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary, as in a standard LM.

In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random. In contrast to denoising auto-encoders, we only predict the masked words rather than reconstructing the entire input.

### Next Sentence Prediction (NSP)

Many important downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI) are based on understanding the relationship between two sentences, which is not directly captured by language modeling. In order to train a model that understands sentence relationships, we pre-train for a binarized next sentence prediction task.

## Training Data and Procedure

BERT was pre-trained on two large text corpora: BooksCorpus (800M words) and English Wikipedia (2,500M words). For Wikipedia, we extract only the text passages and ignore lists, tables, and headers.

The training procedure uses the Adam optimizer with learning rate of 1e-4, beta1 = 0.9, beta2 = 0.999, L2 weight decay of 0.01, learning rate warmup over the first 10,000 steps, and linear decay of the learning rate.

## Fine-tuning Procedure

For fine-tuning, most model hyperparameters are the same as in pre-training, with the exception of the batch size, learning rate, and number of training epochs. The optimal hyperparameter values are task-specific, but we found the following range of possible values to work well across all tasks: Batch size: 16, 32; Learning rate (Adam): 5e-5, 3e-5, 2e-5; Number of epochs: 2, 3, 4.

## Results

BERT achieves state-of-the-art performance on a wide range of NLP benchmarks including GLUE, SQuAD, and SWAG. The key insight is that bidirectional pre-training is more effective than unidirectional pre-training for language understanding tasks.

## Conclusion

BERT demonstrates the importance of bidirectional pre-training for language representations. Using a masked language model enables the use of deep bidirectional representations, in contrast to approaches which use shallow concatenation of independently trained left-to-right and right-to-left models. BERT advances the state of the art for eleven NLP tasks.
