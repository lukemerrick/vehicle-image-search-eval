# Project: Evaluate Image Search

The core idea of this project is to create a difficult evaluation for image search in order to evaluate some recent image + text embedding models (notably MetaCLIP 1.2 and Jina CLIP v2).

## The eval dataset -- Vehicle Image Search

We begin the creation of our evaluation challenged by using a pretrained Imagenet1k classifier to identify a subset of the Flickr30k dataset (see below) that likely contains some kind of vehicle as a prominent element. We then augment this subset of ~2k captioned images with un-captioned Imagenet1k images from vehicle classes. We also downsize the images to make the eval smaller (160px in the smallest dimension).

In the end, we end up with a large set of images (68k) which take up less than 1G of disk space (720M) and are paired with just over 10k queries (which pertain to only a subset of ~2k images from the Flickr30k dataset). By focusing on one domain, we hope to replicate "one in a million" image search without actually requiring a million image embeddings per evaluation (which is costly).

## Results

|         |   MetaCLIP 1.2 |   Jina CLIP v2 |
|:--------|---------------:|---------------:|
| NDCG@10 |          0.511 |          0.384 |
| MAP@10  |          0.467 |          0.340 |
| R@10    |          0.651 |          0.527 |
| P@10    |          0.065 |          0.053 |

## Creating the evaluation data

```shell
# Download Flickr30k.
PYTHONPATH=. python scripts/download_flickr30k.py ./flickr30k

# Extract vehicle images and captions as a retrieval benchmark.
PYTHONPATH=. python scripts/find_and_resize_flickr30k_vehicle_images.py ./flickr30k ./vehicle_image_search

# Authenticate with HuggingFace to download Imagenet1k.
# (Note, you must also accept the terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k)
huggingface-cli login

# Add all vehicle images from Imagenet1k training and validation splits to make search harder.
PYTHONPATH=. python scripts/download_and_resize_imagenet1k_vehicles.py ./vehicle_image_search/images
```


## Literature skim (lit review)


### MetaCLIP

As of 2024-12, MetaCLIP v1.2 is one of the strongest image-text models out there. It's backed by two papers:

- [Altogether: Image Captioning via Re-aligning Alt-text](https://arxiv.org/abs/2410.17251)
  - This is the more recent work. It focuses on creating a powerful image-captioning system that takes in both images and alt-text to generate high-quality, factually-grounded captions (e.g. using alt-text that specifies animal species to caption more correctly rather than hallucinating a species).

- [Demystifying CLIP Data](https://arxiv.org/abs/2309.16671)
  - This is the original work behind MetaCLIP. It presents a pipeline for creating high-quality training data for CLIP models.


#### *Alt*ogether-FT

The Altogether-FT dataset (https://huggingface.co/datasets/activebus/Altogether-FT) is developed using a human-in-the-loop iterative refinement process (generate and then critique via annotation), and it consists of 15k examples of high-quality captions grounded in image + alt-text pairs.

The FAIR team behind the Altogether-FT dataset uses it to finetune a image-embedding-to-caption generation model which in turn is used to generate a large CLIP training dataset with higher-quality image-caption pairs.
- The key caption generation model is a 1.3B OPT text decoder model
- They use a preexisting CLIP model to map images to image embedding vectors
- They translate image embedding vectors into a fixed-size chunk of 40 visual token embeddings via a small "mapping network"
- They train end-to-end on the task of mapping image + alt text into a good caption, as in prior works
- Their dataset allows the model to take advantage of alt-text as well as the image pixels and produce much better captions than prior approaches

#### Demystifying CLIP Data

As indicated by the name, this work demonstrates how to create a high-quality contrastive image-language dataset by leveraging image metadata. They show that balancing over the metadata distribution drives a massive boost in model quality equivalent to dramatically longer training runs on unbalanced data. The original CLIP dataset balances a 400M-pair dataset into 500k groups via looking at image metadata containing one of 500k different query terms mined from Wikipedia bigram frequency, Wikipedia article titles, etc.

The key benefit here is removing a massive fat tail of metadata topics, e.g. only 20k examples with metadata indicating the word "photo" are kept out the 54M-example subset of the 400M examples that contain this word.


#### MetaCLIP overall

Putting the above works together, it seems that the final training dataset consisting of well-balanced data re-labeled with high-quality, highly-descriptive captions from their novel captioning system delivers a 400M-pair dataset of distrbution-balanced images and high-quality descriptive captions. Using the standard CLIP training approach on this dataset yields one of the strongest image-text models produced to date

**Conclusion:** It's a good idea to use the MetaCLIP 1.2 model as a base model for image-text embedding works. Additionally, wiring this model into a captioning system that leverages a mapping model, generative language model and fine-tuning on the *Alt*ogether finetuning dataset is a recipe for creating a very powerful image captioning system.

### Jina CLIP v2

Jina initially focused on developing text embeding models with retrieval as a primary usecase. They expanded into embedding models for image and text retrieval with a focus on keeping text-only retrieval quality high (much higher than delivered by typical CLIP models).

The Jina CLIP v2 paper covers their approach to training a high-quality embedding model with a focus on search. In particular, this technical report serves as a good demonstration of different standardized evaluations.

Looking at evaluation datasets, Jina CLIP v2 uses MTEB retrieval and STS tasks for text-to-text embedding operations, while relying on several different image-text datasets for image-to-text (I-T) and text-to-image (T-I) retrieval quality assessment. In particular, [Flicker30k](https://huggingface.co/datasets/nlphuji/flickr30k?row=12) looks like it has slightly more detailed captions which could be used as complicated queries (e.g. "A black dog and a white dog with brown spots are staring at each other in the street" and "Man wearing a blue and white outfit, holding a broom, with a traditional Asian architecture in the background"), while [MS COCO Captions](https://huggingface.co/datasets/sentence-transformers/coco-captions?row=47) appears more simplistic (e.g. "A kitchen has red bricks lining the counter" and "A man getting ready to surf as lookers walk by"). 

**Conclusion:** For now let's look at Flickr30k captions as queries, but for retrieval evaluation we should include more that 30k example images to make retrieval more difficult. Idea: Embed and cluster Flicker30k, then create a subsample of the data which focuses on semantically similar topics, then expand with additional hard-negative images to create a tougher search evaluation.

## Dev setup

I threw together a few "keep your Python clean" tools below. This is probably still a work in progress, and it's probably a lot of premature optimization, too.

###  Pre-commit

[UV precommit docs](https://docs.astral.sh/uv/guides/integration/pre-commit/)
[ruff precommit docs](https://github.com/astral-sh/ruff-pre-commit)
[pyright precommit docs](https://github.com/RobertCraigie/pyright-python?tab=readme-ov-file#pre-commit)

See the `pyproject.toml` for ruff config.