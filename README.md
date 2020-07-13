# CTDeepRot
A tool for automatic rotation of spinal CT data into a standardized (HFS) patient position. The proposed method is based on the prediction of rotation angle with CNN, and it achieved nearly perfect results with an accuracy of 99.55 % and in a very short time (in the order of units of seconds per scan). We provide implementations with easy to use an example for both Matlab and Python (PyTorch), which can be used, for example, for automatic rotation correction of VerSe2020 challenge data.

Code can be executed by "example" script in "example_prediction" folder for both, Matalb and Python. Just change the variable "file_name", which should be full path to your mhd/raw file.
