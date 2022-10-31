# CNN-for-modulation-recognition-based-on-FPGA

Here are Xueyuan’s thesis work. These codes can realize:
1. Convert I&Q **RadioML2016.10a** dataset to constellation plot dataset: 
   `create_const_all_points.ipynb` and `create_constellation_discarded.ipynb`
2. Training neural network to obtain network parameters: `CNNfor_mod_recog.py`
3. Extract weight and bias matrixes: `extract_weibias.py`
4. Convert Tensorflow parameters to format that VHDL can use (float to int): `transform_weightbias.py`
5. FPGA (VHDL) non-timed CNN for implementing modulation classification: folder `vhdl_non-timed_network`
6. FPGA (VHDL) hardware synthesizable CNN for implementing modulation classification: folder `vhdl_hardware_synthesizable_network`

This work also corresponds to our paper, if you want to use this repository please cite:
```
@INPROCEEDINGS{8922403,
  author={LIU, Xueyuan and SHANG, Jing and Leong, Philip H.W. and LIU, Cheng},
  booktitle={2019 22nd International Conference on Electrical Machines and Systems (ICEMS)}, 
  title={Modulation recognition using an FPGA-based convolutional neural network}, 
  year={2019},
  pages={1-6},
  doi={10.1109/ICEMS.2019.8922403}}
```
