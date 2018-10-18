# CNN-for-modulation-recognition-based-on-FPGA
Here are Xueyuan’s thesis work. These codes can realize:
1.	Convert I&Q dataset to constellation plot dataset.
2.	Training neural network to obtain network parameters
3.	Extract weight and bias matrixes
4.	VHDL CNN network for implementing modulation classification
Read me 
Here are Xueyuan’s thesis work. These codes can realize:

1. Convert I&Q RadioML2016.10a dataset to constellation plot dataset: 
   `create_const_all_points.ipynb and create_constellation_discarded.ipynb`

Training neural network to obtain network parameters: CNNfor_mod_recog.py

Extract weight and bias matrixes: extract_weibias.py
Convert Tensorflow parameters to format that VHDL can use: transform_weightbias.py
VHDL CNN network for implementing modulation classification: several files in VHDL folder.
