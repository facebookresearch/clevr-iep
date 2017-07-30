# inferring-and-executing

This is the code for the paper

 **<a href="https://arxiv.org/abs/1705.03633">Inferring and Executing Programs for Visual Reasoning</a>**
 <br>
 <a href='http://cs.stanford.edu/people/jcjohns/'>Justin Johnson</a>,
 <a href='http://home.bharathh.info/'>Bharath Hariharan</a>,
 <a href='https://lvdmaaten.github.io/'>Laurens van der Maaten</a>,
 <a href='http://cs.stanford.edu/~jhoffman/'>Judy Hoffman</a>,
 <a href='http://vision.stanford.edu/feifeili/'>Fei-Fei Li</a>,
 <a href='http://larryzitnick.org/'>Larry Zitnick</a>,
 <a href='http://www.rossgirshick.info/'>Ross Girshick</a>
 <br>
 To appear at [ICCV 2017](http://iccv2017.thecvf.com/)

<div align="center">
  <img src="https://github.com/facebookresearch/clevr-iep/blob/master/img/system.png" width="450px">
</div>

If you find this code useful in your research then please cite

```
@inproceedings{johnson2017inferring,
  title={Inferring and Executing Programs for Visual Reasoning},
  author={Johnson, Justin and Hariharan, Bharath and van der Maaten, Laurens and Hoffman, Judy
          and Fei-Fei, Li and Zitnick, C Lawrence and Girshick, Ross},
  booktitle={ICCV},
  year={2017}
}
```

# Setup

All code was developed and tested on Ubuntu 16.04 with Python 3.5.

You can set up a virtual environment to run the code like this:

```bash
virtualenv -p python3 .env       # Create virtual environment
source .env/bin/activate         # Activate virtual environment
pip install -r requirements.txt  # Install dependencies
echo $PWD > .env/lib/python3.5/site-packages/iep.pth # Add this package to virtual environment
# Work for a while ...
deactivate # Exit virtual environment
```

# Pretrained Models
You can download and unzip the pretrained models by running `bash scripts/download_pretrained_models.sh`;
the models will take about 1.1 GB on disk.

We provide two sets of pretrained models:
- The models in `models/CLEVR` were trained on the CLEVR dataset; these were used to make Table 1 in the paper.
- The models in `models/CLEVR-Humans` were first trained on CLEVR and then finetuned on the CLEVR-Humans dataset;
   these models were used to make Table 3 in the paper.

# Running models

You can easily run any of the pretrained models on new images and questions. As an example, we will run several
models on the following example image from the CLEVR validation set:

<div align='center'>
 <img src='https://github.com/facebookresearch/clevr-iep/blob/master/img/CLEVR_val_000013.png'>
</div>

After downloading the pretrained models, you can use the pretrained model to answer questions about this image with
the following command:

```bash
python scripts/run_model.py \
  --program_generator models/CLEVR/program_generator_18k.pt \
  --execution_engine models/CLEVR/execution_engine_18k.pt \
  --image img/CLEVR_val_000013.png \
  --question "Does the small sphere have the same color as the cube left of the gray cube?"
```

This will print the predicted answer, as well as the program that the model used to produce the answer.
For the example command we get the output:

```bash
Question: "Does the small sphere have the same color as the cube left of the gray cube?"
Predicted answer:  yes

Predicted program:
equal_color
query_color
unique
filter_shape[sphere]
filter_size[small]
scene
query_color
unique
filter_shape[cube]
relate[left]
unique
filter_shape[cube]
filter_color[gray]
scene
```

# Training

The procedure for training your own models [is described here](TRAINING.md).
