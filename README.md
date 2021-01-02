pip3 uninstall tensorflow tensorflow-estimator && pip3 install tensorflow==1.8


https://www.tensorflow.org/install/pip?hl=ko#raspberry-pi

https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi


REQUIRED_PACKAGES = ['Pillow>=1.0', 'Matplotlib>=2.1', 'Cython>=0.28.1']

setup(
    name='object_detection',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('object_detection')],
    description='Tensorflow Object Detection Library',
)


virtual machin

http://raspberrypi-aa.github.io/session4/venv.html


venv 
python3 -m venv ~/myvenv
https://www.techcoil.com/blog/how-to-use-python-3-virtual-environments-to-run-python-3-applications-on-your-raspberry-pi/
