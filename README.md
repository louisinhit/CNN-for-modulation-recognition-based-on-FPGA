# CNN-for-modulation-recognition-based-on-FPGA

Here are Xueyuan’s thesis work. These codes can realize:
1. Convert I&Q **RadioML2016.10a** dataset to constellation plot dataset: 
   `create_const_all_points.ipynb` and `create_constellation_discarded.ipynb`
2. Training neural network to obtain network parameters: `CNNfor_mod_recog.py`
3. Extract weight and bias matrixes: `extract_weibias.py`
4. Convert Tensorflow parameters to format that VHDL can use: `transform_weightbias.py`
5. VHDL CNN network for implementing modulation classification: several files in VHDL folder.
