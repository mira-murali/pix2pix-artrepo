# pix2pix-artrepo
Artistic representations using pix2pix GANs
### Preparing the Data
Before running the code, ensure that you have the data in the following format:

images/

&nbsp;&nbsp;&nbsp;&nbsp;blurred_images/

&nbsp;&nbsp;&nbsp;&nbsp;hed/

&nbsp;&nbsp;&nbsp;&nbsp;orig_images

Please make sure you use these names for the data.

Then you can run:

`python prepare_data.py images/ images/
`

The first command line argument is the dataroot, i.e., it tells the code where all of the data is stored.

The second command line argument is for where you want to save the train-validation split. In the above example, the train-val images are also saved in the same directory. You are free to use a different directory. The train-val split will be stored in the following format:

images/

&nbsp;&nbsp;&nbsp;&nbsp;A/

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;train/

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;blurred_images/

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; hed/

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test/

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;blurred_images/

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;hed/

&nbsp;&nbsp;&nbsp;&nbsp;B/

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;train/

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test/

You will then set DATAROOT in `hyperparameters.py` to be the path to 'images'.

If your color and black-and-white representations are not called 'blurred_images' and 'hed' respectively, remember to change them in `hyperparameters.py` (COLOR_NAME and BW_NAME).

NOTE: You would also have to change the names manually in `prepare_data.py` as well BEFORE you start running any of the code.

### Representations
UPDATE: No longer need to change the visdom `__init__.py` function. Skip to the training section.

Because we are using a 4-channel image as our A representation, this causes problems while using visdom. The right solution would be to switch to tensorboard or rewrite the code to accept 4-channel images but a quick fix is to change the `__init__.py` of visdom.

Assuming you have visdom in your virtual environment, you will have to navigate to your virtual environment directory, and cd into python site-packages, and into visdom. You will find `__init__.py` there. Under the function 'images', you will have to make some minor changes which has been specified under a similar function 'images' in `visdom_init.py`.

An easier alternative is to simply copy all the contents of `visdom_init.py` to `__init__.py`

### Training

To train the model, type this in the command line:

`python train.py
`

You can specify the number of epochs, the experiment name, the batch size and other hyperparameters in `hyperparameters.py`. This will also perform validation on 10 images at the end of each epoch.

### Testing
To test the model, simply type this in the command line:

`python test.py --model pix2pix
`

It is crucial that you set the model name as 'pix2pix' in the command line for it to test without errors.