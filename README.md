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



## Dev setup

I threw together a few "keep your Python clean" tools below. This is probably still a work in progress, and it's probably a lot of premature optimization, too.

###  Pre-commit

[UV precommit docs](https://docs.astral.sh/uv/guides/integration/pre-commit/)
[ruff precommit docs](https://github.com/astral-sh/ruff-pre-commit)
[pyright precommit docs](https://github.com/RobertCraigie/pyright-python?tab=readme-ov-file#pre-commit)

See the `pyproject.toml` for ruff config.