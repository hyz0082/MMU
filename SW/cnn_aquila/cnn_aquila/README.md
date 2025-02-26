# Guide of our CNN library
## compile binary for Gem5 SE mode
```
make [alexnet/resnet18/resnet50/ultranet]
```
After make, we will have the corresponding executable binary file for Gem5 simulator.
## convert weight and preprocessed image from Pytorch
```
cd python
python [alexnet.py/resnet18.py/resnet50.py]
```
after execute python script, we will have the weight and preprocessed image in IEEE754 format, we have to copy these files to the suitable directory.

Take Samoyed.jpg as ths input of python script and alexnet for example:
```
cp Samoyed.bin ../input_file
cp weights/alexnet_Weight ../weights
```
After doing all things above, we can start our simulation using the provided configuration script:
```
cd ..
<gem5_path>/MESI_Two_Level/gem5.opt ./gem5/test_alexnet.py
```