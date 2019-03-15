# pix2pix-artrepo
Artistic representations using pix2pix GANs

### Data Structure
Ensure that your data is stored in the following format:

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
### Running the code
Because we are using a 4-channel image as our A representation, this causes problems while using visdom. The right solution would be to switch to tensorboard or rewrite the code to accept 4-channel images but a quick fix is to change the `__init__.py` of visdom.

Assuming you have visdom in your virtual environment, you will have to navigate to your virtual environment directory, and cd into python site-packages, and into visdom. You will find `__init__.py` there. Under the function 'images', you will have to make some minor changes which has been specified under a similar function 'images' in `visdom_init.py`.

An easier alternative is to simply copy all the contents of `visdom_init.py` to `__init__.py`

To train the model, type this in the command line:

`python train.py
`
You can specify the number of epochs, the experiment name, the batch size and other hyperparameters in `hyperparameters.py`
