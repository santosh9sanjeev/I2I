"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from models.utils import SimpleLogger, eval_loader
import os
import torch
from data import create_dataset, get_test_loaders
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    
    test_loader_A, test_loader_B = get_test_loaders(opt)
    fix_a = torch.stack([test_loader_A.dataset[i]['A'] for i in range(opt.display_size)]).cuda()  # fixed test data
    fix_b = torch.stack([test_loader_B.dataset[i]['A'] for i in range(opt.display_size)]).cuda()
    test_logger = SimpleLogger(os.path.join(opt.results_dir, 'test.txt'))
    
    rmse = []
    acc5 = []
    acc10 =[]
    FID = []
    KID = []

    for epoch_i in range(5,125,5):
        print(epoch_i)
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt, epoch_i)               # regular setup: load and print networks; create schedulers
        # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
        # if opt.load_iter > 0:  # load_iter is 0 by default
        #     web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
        # print('creating web directory', web_dir)
        # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))    
        
        if opt.eval:
            model.eval()
        # for i, data in enumerate(dataset):
        #     if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #         break
        # model.set_input(data)  # unpack data from data loader
        # model.test()           # run inference
        # visuals = model.get_current_visuals()  # get image results
        # img_path = model.get_image_paths()     # get image paths
                        
                        
        eval_dict = eval_loader(model, test_loader_A, test_loader_B, opt.results_dir, opt)
        test_logger.log(epoch_i, 125, eval_dict, verbose=True)
        rmse.append(eval_dict['/rmse'])
        acc5.append(eval_dict['/acc@5'])
        acc10.append(eval_dict['/acc@10'])
        FID.append(eval_dict['FID'])
        KID.append(eval_dict['KID'])
        
        #     if i % 5 == 0:  # save images to an HTML file
        #         print('processing (%04d)-th image... %s' % (i, img_path))
        #     save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        # webpage.save()  # save the HTML
        plt.plot(range(5, epoch_i+1, 5), rmse, label='RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss')
        plt.title('RMSE vs. Epoch')
        plt.legend()
        filepath = os.path.join(opt.results_dir, 'rmse.png')

        plt.savefig(filepath)
        plt.close()
        
        # Plot validation loss vs. epoch
        plt.plot(range(5, epoch_i+1, 5), acc5, label='acc5')
        plt.xlabel('Epoch')
        plt.ylabel('acc5')
        plt.title('acc5 vs. Epoch')
        plt.legend()
        filepath = os.path.join(opt.results_dir, 'acc5_plot.png')

        plt.savefig(filepath)
        plt.close()
        
        # Plot test loss vs. epoch
        plt.plot(range(5, epoch_i+1, 5), acc10, label='acc10')
        plt.xlabel('Epoch')
        plt.ylabel('acc10')
        plt.title('acc10 vs. Epoch')
        plt.legend()
        filepath = os.path.join(opt.results_dir, 'acc10_plot.png')

        plt.savefig(filepath)
        plt.close()
        
        # Plot test accuracy vs. epoch
        plt.plot(range(5, epoch_i+1, 5), FID, label='FID')
        plt.xlabel('Epoch')
        plt.ylabel('FID')
        plt.title('FID vs. Epoch')
        plt.legend()
        filepath = os.path.join(opt.results_dir, 'FID_plot.png')

        plt.savefig(filepath)
        plt.close()


        plt.plot(range(5, epoch_i+1, 5), KID, label='KID')
        plt.xlabel('Epoch')
        plt.ylabel('KID')
        plt.title('KID vs. Epoch')
        plt.legend()
        filepath = os.path.join(opt.results_dir, 'KID_plot.png')

        plt.savefig(filepath)
        plt.close()
