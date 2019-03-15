"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from options.val_options import ValOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import os 
from util import html
from util.visualizer import save_images
from tqdm import tqdm

if __name__ == '__main__':
    train_opt = TrainOptions().parse()   # get training options
    train_dataset = create_dataset(train_opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset_size = len(train_dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % train_dataset_size)

    model = create_model(train_opt)      # create a model given opt.model and other options
    model.setup(train_opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(train_opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    val_opt = ValOptions().parse()
    val_dataset = create_dataset(val_opt)
    
    for epoch in range(train_opt.epoch_count, train_opt.niter + train_opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.isTrain = True
        for i, data in tqdm(enumerate(train_dataset)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % train_opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += train_opt.batch_size
            epoch_iter += train_opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % train_opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % train_opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % train_opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / train_opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if train_opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / train_dataset_size, losses)

            if total_iters % train_opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if train_opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % train_opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        
        web_dir = os.path.join(val_opt.results_dir, val_opt.name, '%s_%s' % (val_opt.phase, train_opt.epoch))  # define the website directory
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (val_opt.name, val_opt.phase, train_opt.epoch))
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        if val_opt.eval:
            model.eval()
        for i, data in enumerate(val_dataset):
            if i >= val_opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=val_opt.aspect_ratio, width=val_opt.display_winsize)
        webpage.save()  # save the HTML
            

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, train_opt.niter + train_opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
