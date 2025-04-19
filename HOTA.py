#---------------------------------#
# class_name_to_class_id = {'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,
#                           'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8, 'occluder': 9,
#                           'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12, 'crowd': 13}
#---------------------------------#clear
import os
import sys
import json
import argparse
from configs.config import get_config 
sys.path.append(os.path.join('thirdparty','TrackEval'))
import trackeval 
from multiprocessing import freeze_support


def parse_args():
    parser = argparse.ArgumentParser(description='HOTA')
    parser.add_argument('--dataset', type=str, default='MOT17', help='config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset_name = args.dataset
    cfg   = get_config()
    with open(cfg.JSON_PATH,'r') as f:
        data_json = json.load(f)
    freeze_support()

    eval_config = {
        'USE_PARALLEL':False,
        'NUM_PARALLEL_CORES':8,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'DISPLAY_LESS_PROGRESS': True,

        'OUTPUT_SUMMARY': True,
        'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
        'OUTPUT_DETAILED': False,
        'PLOT_CURVES': False,
    }
    evaluator = trackeval.Evaluator(eval_config)

    metrics_list = [trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(),trackeval.metrics.Identity()]


    dataset_config = {

        # 真值文件路径设置
        'GT_FOLDER':data_json['Trackeval']['GT_FOLDER'],
        # 跟踪器输出结果文件路径设置
        'TRACKERS_FOLDER':data_json['Trackeval']['TRACKERS_FOLDER'],
        'SKIP_SPLIT_FOL': False,  # 自用数据所需设置为True
        'BENCHMARK':dataset_name,
        'SPLIT_TO_EVAL':data_json['valid_seq'][dataset_name]['Trackeval']['SPLIT_TO_EVAL'],
    }

    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    raw_results , messages = evaluator.evaluate(dataset_list,metrics_list)
    
    
    record_metrics_list = {
        'HOTA':['HOTA','DetA','AssA'],
        'Identity':['IDF1','IDR','IDP'],
        'CLEAR':['MOTA','MOTP'],
    }
    for type, tracker in raw_results.items():
        for tracker_name , metrics_per_seq in tracker.items():
            print(f"Tracker:{tracker_name}")
            for cls ,number_per_metrics in metrics_per_seq['COMBINED_SEQ'].items():
                for metrics,index_list in record_metrics_list.items():
                    for index in index_list:
                        print(f"{index}(%):[{number_per_metrics[metrics][index].mean() * 100 :.2f}]")