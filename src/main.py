import logging.config
import sys
from datetime import datetime
from logging import Logger

import torch
import yaml

from lib.helpers.utills import check_data_path, model_path_handler
from lib.sentence_transformers import GettySentenceTransformer
from preprocessing.preprocessing import *
from training.train import train

with open('./logs/conf.yaml') as stream:
    yaml_log_config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(yaml_log_config)

process_logger: Logger = logging.getLogger('pipeline')
net_logger: Logger = logging.getLogger('network')

if __name__ == '__main__':

    with open("./config.yaml") as config_file:
        cfg = yaml.load(config_file)
    process_logger.info('--------------')
    process_logger.info(f"Configurations: {cfg}")
    
    if cfg['preprocessing']['inputpath_origin'] == 'local':
        processing_data(cfg['preprocessing']['Input_file_directory_path'],
                    cfg['preprocessing']['Output_directory_path'])  # calling preprocessing function
    else:
        processing_s3data(cfg['preprocessing']['Output_directory_path'],
                       cfg['preprocessing']['aws_secret'],cfg['preprocessing']['aws_id'],cfg['preprocessing']['bucket_name'],cfg['preprocessing']['folder_name'])  # calling preprocessing function
        
    if not check_data_path(cfg['training']['training_data_path'],cfg['training']['inputpath_origin']):
        process_logger.info('training data path is not correct')
        sys.exit()

    if not check_data_path(cfg['training']['test_data_path'],cfg['training']['inputpath_origin']):
        process_logger.info('test data path is not correct')
        sys.exit()

    if not cfg['training']['multilingual']:
        process_logger.info("bert-base-nli-stsb-mean-tokens model will be used")
        model_name = 'bert-base-nli-stsb-mean-tokens'
        cfg['training']['trained_model_path'] = model_path_handler(model_name, cfg['training']['trained_model_path'])

        cfg['training']['model_save_path'] = cfg['training'][
                                                 'model_save_path'] + '/training_stsbenchmark_continue_training-' + model_name + '-' + datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S")
    else:
        process_logger.info("stsb-xlm-r-multilingual model will be used")
        model_name = 'stsb-xlm-r-multilingual'
        cfg['training']['trained_model_path'] = model_path_handler(model_name, cfg['training']['trained_model_path'])

        cfg['training']['model_save_path'] = cfg['training'][
                                                 'model_save_path'] + '/training_stsmultilingual_continue_training-' + model_name + '-' + datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S")

    device = None
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if cfg['training']['gpu_id'] < 0:
            process_logger.info("Device id not correct, default device is used")
            device = torch.device('cuda:0')
            process_logger.info("Use pytorch device: {}".format(device))
        elif cfg['training']['gpu_id'] >= n_gpu:
            process_logger.info("Device id not correct, default device is used")
            device = torch.device('cuda:0')
            process_logger.info("Use pytorch device: {}".format(device))
        else:
            device = torch.device('cuda:{}'.format(cfg['training']['gpu_id']))
            process_logger.info("Use pytorch device: {}".format(device))

        if cfg['training']['multi_gpus']:
            device = torch.device('cuda:{}'.format(0))
            process_logger.info("Use pytorch device: {}, as multi-gpus is enable".format(device))

    # Load a pre-trained sentence transformer model
    model = GettySentenceTransformer(cfg['training']['trained_model_path'], device=device)

    train(model, cfg)

