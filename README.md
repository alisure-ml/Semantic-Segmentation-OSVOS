### OSVOS: One-Shot Video Object Segmentation


### Data: `Dataset`
> dataset.py

* Download [DAVIS 2016: DAVIS-data]((https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip) ) and unzip 
in YOU/PATH config in `OSVOSDemo.davis_root`


### Net
> osvos.py

* Unzip [vgg_16_2016_08_28.tar.gz on Imagenet](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) 
in `models/` config in `OSVOSParentDemo.imagenet_ckpt`


### Train: `OSVOSParentDemo`
> osvos_parent_demo.py

* Unzip [OSVOS_parent_model](https://data.vision.ee.ethz.ch/csergi/share/OSVOS/OSVOS_parent_model.zip)(55MB) 
in `models/OSVOS_parent` config in `OSVOSParentDemo.model_result_path` and `OSVOSDemo.parent_model`


### Test: `OSVOSDemo`
> osvos_demo.py

* Unzip [OSVOS_pre-trained_models](https://data.vision.ee.ethz.ch/csergi/share/OSVOS/OSVOS_pre-trained_models.zip)(2.2GB) 
in `models/OSVOS_demo` config in `OSVOSDemo.model_result_path`


### Reference
* [scaelles/OSVOS-TensorFlow](https://github.com/scaelles/OSVOS-TensorFlow)
* [Author project page](http://www.vision.ee.ethz.ch/~cvlsegmentation/osvos)
* [OSVOS-caffe](https://github.com/kmaninis/OSVOS-caffe)
* [OSVOS-PyTorch](https://github.com/kmaninis/OSVOS-PyTorch)


### Citation:
	@Inproceedings{Cae+17,
	  Title          = {One-Shot Video Object Segmentation},
	  Author         = {S. Caelles and K.K. Maninis and J. Pont-Tuset and L. Leal-Taix\'e and D. Cremers and L. {Van Gool}},
	  Booktitle      = {Computer Vision and Pattern Recognition (CVPR)},
	  Year           = {2017}
	}
	