from argparse import ArgumentParser
from src.ahs.utils.load_config import load_config
from src.ahs.training.infer import infer


if __name__== '__main__':
    """Parse CLI arguments for infer"""
    parser = ArgumentParser(description='args for training')
    parser.add_argument("--img_path", type=str, default='images/original_img.jpg', help='path to test image')
    parser.add_argument("--checkpoint", type=str, default='experiments/outputs/best.pt', help='path to save checkpoint')
    parser.add_argument("--img_size", type=int, default=480, help='image size')

    args = parser.parse_args()
    
    # load dataset, augmentation and training config
    infer_cfg = load_config('config/infer.yaml')
    
    # train merge
    infer_cfg['img_path'] = args.img_path or infer_cfg['img_path']
    infer_cfg['checkpoint'] = args.checkpoint or infer_cfg['checkpoint']
    infer_cfg['img_size'] = args.img_size or infer_cfg['img_size']
    
    # call main train func
    infer(infer_cfg['img_path'], infer_cfg['checkpoint'], infer_cfg['img_size'], infer_cfg['num_cls'], infer_cfg['score_thresh'])