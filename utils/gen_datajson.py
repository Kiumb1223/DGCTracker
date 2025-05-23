import os
import sys
import json 
import shutil
import argparse
from pathlib import Path

_ROOT =Path(__file__).parent.parent
sys.path.append(os.path.join(_ROOT,'configs'))
from config import get_config  # type: ignore

def parse_args():
    parser = argparse.ArgumentParser(description='Generate data.json')
    parser.add_argument('--data_construct_type', type=str, default='mot_challenge')
    parser.add_argument('--datasets', nargs='+', default=['MOT17'])
    parser.add_argument('--save_dataset_dir', type=str, default='datasets/eval_datasets')
    parser.add_argument('--save_json_dir', type=str, default='configs')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = get_config()

    data_construct_type = args.data_construct_type
    save_dataset_dir = args.save_dataset_dir
    save_json_dir = args.save_json_dir

    seqmap_dict = {}
    save_json = {
        'train_seq':{},
        'valid_seq':{},
        'Trackeval':{
            'GT_FOLDER':os.path.join(save_dataset_dir,'gt',data_construct_type),
            'TRACKERS_FOLDER':os.path.join(save_dataset_dir,'trackers',data_construct_type),
            'SKIP_SPLIT_FOL':True,
        }
    }

    which_to_train_list = args.datasets
    # which_to_train_list = ['MOT17','MOT20']
    # which_to_train_list = ['MOT20']
    # save_json_path      = 'configs/train_MOT20.json'
    
    
    for dataset_name in which_to_train_list:
        save_json_path = save_json_dir + os.sep + dataset_name +'.json'
        save_json['train_seq'][dataset_name] = {'seq_name':[],'start_frame':[],'end_frame':[],}
        save_json['valid_seq'][dataset_name] = {'seq_name':[],'start_frame':[],'end_frame':[],
            'Trackeval':{ # plz SEE https://github.com/JonathonLuiten/TrackEval
            'SPLIT_TO_EVAL':None,
            
        }}
        if dataset_name in ['MOT17','MOT20','MOT16']:
            seqmap_dict[dataset_name] = {
                'path' : None,
                'items':['name\n',]
            }
            for seq in os.listdir(os.path.join(cfg.DATA_DIR,dataset_name,'train')):
                if not os.path.isdir(os.path.join(cfg.DATA_DIR,dataset_name,'train',seq)):
                    continue
                seq_path = os.path.join(cfg.DATA_DIR,dataset_name,'train',seq)
                ini_path = os.path.join(seq_path,'seqinfo.ini')
                with open(ini_path,'r') as f:
                    lines_split = [ l.split('=') for l in f.read().splitlines()[1:]]
                    info_dict  = dict(s for s in lines_split if isinstance(s,list) and len(s) == 2)
                
                save_json['train_seq'][dataset_name]['seq_name'].append(seq) # half-frame data for train 
                save_json['train_seq'][dataset_name]['start_frame'].append(1)
                save_json['train_seq'][dataset_name]['end_frame'].append(int(info_dict['seqLength'])-100)

                save_json['valid_seq'][dataset_name]['seq_name'].append(seq) # half-frame data for validation 
                save_json['valid_seq'][dataset_name]['start_frame'].append(int(info_dict['seqLength'])-100 + 1 )
                save_json['valid_seq'][dataset_name]['end_frame'].append(int(info_dict['seqLength']))
                
                suffix     = 'last100'
                seqmap_name= dataset_name+'-'+suffix
                save_json['valid_seq'][dataset_name]['Trackeval']['SPLIT_TO_EVAL']  = suffix
                gt_seqmap_folder   = os.path.join(save_json['Trackeval']['GT_FOLDER'],'seqmaps')
                gt_dataname_folder = os.path.join(save_json['Trackeval']['GT_FOLDER'],seqmap_name)
                gt_dataname_folder = os.path.join(save_json['Trackeval']['GT_FOLDER'],seqmap_name)
                track_seqname_folder = os.path.join(save_json['Trackeval']['TRACKERS_FOLDER'],seqmap_name)
                os.makedirs(gt_seqmap_folder,exist_ok=True)
                os.makedirs(track_seqname_folder,exist_ok=True)

                seqmap_dict[dataset_name]['path'] = os.path.join(gt_seqmap_folder,f'{seqmap_name}.txt')
                seqmap_dict[dataset_name]['items'].append(seq + '\n')

                # # for seqmap
                # with open(os.path.join(gt_seqmap_folder,f'{seqmap_name}.txt'),'a+') as f:
                #     if os.path.getsize(os.path.join(gt_seqmap_folder,f'{seqmap_name}.txt')) == 0:
                #         f.write('name\n')
                #     f.write(seq+'\n')
                
                # move some necessary file for evalution
                gt_seq_folder = os.path.join(gt_dataname_folder,seq)
                gt_seq_subfolder = os.path.join(gt_seq_folder,'gt')
                os.makedirs(gt_seq_subfolder,exist_ok=True)
                with open(os.path.join(gt_seq_folder,'seqinfo.ini'),'w') as f:
                    f.write(
                        f"[Sequence]\n" +
                        f"name={info_dict['name']}\n" +
                        f"imDir={info_dict['imDir']}\n" +
                        f"frameRate={info_dict['frameRate']}\n" +
                        f"seqLength=100 \n" +
                        f"imWidth={info_dict['imWidth']}\n" +
                        f"imHeight={info_dict['imHeight']}\n" +
                        f"imExt={info_dict['imExt']}\n"
                    )

                det_path = os.path.join(seq_path,'det','det.txt')
                gt_path = os.path.join(seq_path,'gt','gt.txt')
                output_det_path = os.path.join(gt_seq_folder,'det')
                output_gt_path = os.path.join(gt_seq_folder,'gt')
                output_img_path = os.path.join(gt_seq_folder,'img1')
                os.makedirs(output_det_path,exist_ok=True)
                os.makedirs(output_gt_path,exist_ok=True)
                os.makedirs(output_img_path,exist_ok=True)

                with open(output_det_path+os.sep + 'det.txt','w') as f: 
                    with open(det_path,'r') as f1:
                        lines = f1.readlines()
                    for line in lines:
                        frame = int(line.split(',')[0])
                        if int(info_dict['seqLength'])-100+ 1 <= frame <= int(info_dict['seqLength']):
                            cols = line.strip().split(',')
                            cols[0] = str(frame - int(info_dict['seqLength']) +100)
                            new_line = ",".join(cols) + "\n"
                            f.write(new_line)
                with open(output_gt_path+os.sep + 'gt.txt','w') as f: 
                    with open(gt_path,'r') as f1:
                        lines = f1.readlines()
                    for line in lines:
                        frame = int(line.split(',')[0])
                        if int(info_dict['seqLength'])-100 + 1 <= frame <= int(info_dict['seqLength']):
                            cols = line.strip().split(',')
                            cols[0] = str(frame - int(info_dict['seqLength']) +100)
                            new_line = ",".join(cols) + "\n"
                            f.write(new_line)
                
                #---------------------------------#
                #  copy files to specified folder and rename them 
                #---------------------------------#
                cnt = 1 
                for i in range(int(info_dict['seqLength']) - 100 + 1,int(info_dict['seqLength'])+1 ):
                    img_path = os.path.join(seq_path,'img1',f'{i:06d}.jpg')
                    new_img_path = os.path.join(output_img_path,f'{cnt:06d}.jpg')
                    shutil.copy(img_path,new_img_path)
                    cnt+=1
        elif dataset_name in ['DanceTrack']:
            seqmap_dict[dataset_name] = {
                'path' : None,
                'items':['name\n',]
            }

            for train_seq in os.listdir(os.path.join(cfg.DATA_DIR,dataset_name,'train')):
                if not os.path.isdir(os.path.join(cfg.DATA_DIR,dataset_name,'train',train_seq)):
                    continue
                seq_path = os.path.join(cfg.DATA_DIR,dataset_name,'train',train_seq)
                ini_path = os.path.join(seq_path,'seqinfo.ini')
                with open(ini_path,'r') as f:
                    lines_split = [ l.split('=') for l in f.read().splitlines()[1:]]
                    info_dict  = dict(s for s in lines_split if isinstance(s,list) and len(s) == 2)

                save_json['train_seq'][dataset_name]['seq_name'].append(train_seq)
                save_json['train_seq'][dataset_name]['start_frame'].append(1)
                save_json['train_seq'][dataset_name]['end_frame'].append(int(info_dict['seqLength']))          

            for valid_seq in os.listdir(os.path.join(cfg.DATA_DIR,dataset_name,'val')):
                if not os.path.isdir(os.path.join(cfg.DATA_DIR,dataset_name,'val',valid_seq)):
                    continue
                seq_path = os.path.join(cfg.DATA_DIR,dataset_name,'val',valid_seq)
                ini_path = os.path.join(seq_path,'seqinfo.ini')
                with open(ini_path,'r') as f:
                    lines_split = [ l.split('=') for l in f.read().splitlines()[1:]]
                    info_dict  = dict(s for s in lines_split if isinstance(s,list) and len(s) == 2)

                save_json['valid_seq'][dataset_name]['seq_name'].append(valid_seq)
                save_json['valid_seq'][dataset_name]['start_frame'].append(1)
                save_json['valid_seq'][dataset_name]['end_frame'].append(int(info_dict['seqLength']))

                suffix     = 'val'
                seqmap_name= dataset_name+'-'+suffix
                save_json['valid_seq'][dataset_name]['Trackeval']['SPLIT_TO_EVAL']  = suffix

                gt_seqmap_folder   = os.path.join(save_json['Trackeval']['GT_FOLDER'],'seqmaps')
                gt_dataname_folder = os.path.join(save_json['Trackeval']['GT_FOLDER'],seqmap_name)
                track_seqname_folder = os.path.join(save_json['Trackeval']['TRACKERS_FOLDER'],seqmap_name)
                os.makedirs(gt_seqmap_folder,exist_ok=True)
                os.makedirs(track_seqname_folder,exist_ok=True)

                seqmap_dict[dataset_name]['path'] = os.path.join(gt_seqmap_folder,f'{seqmap_name}.txt')
                seqmap_dict[dataset_name]['items'].append(valid_seq + '\n')

                # for seqmap
                # with open(os.path.join(gt_seqmap_folder,f'{seqmap_name}.txt'),'a+') as f:
                #     if os.path.getsize(os.path.join(gt_seqmap_folder,f'{seqmap_name}.txt')) == 0:
                #         f.write('name\n')
                #     f.write(valid_seq+'\n')
                
                # move some necessary file for evalution
                gt_seq_folder = os.path.join(gt_dataname_folder,valid_seq)
                gt_seq_subfolder = os.path.join(gt_seq_folder,'gt')
                os.makedirs(gt_seq_subfolder,exist_ok=True)
                shutil.copy(os.path.join(seq_path,'gt','gt.txt'),os.path.join(gt_seq_subfolder,'gt.txt'))
                with open(os.path.join(gt_seq_folder,'seqinfo.ini'),'w') as f:
                    f.write(
                        f"[Sequence]\n" +
                        f"name={info_dict['name']}\n" +
                        f"imDir={info_dict['imDir']}\n" +
                        f"frameRate={info_dict['frameRate']}\n" +
                        f"seqLength={info_dict['seqLength']}\n" +
                        f"imWidth={info_dict['imWidth']}\n" +
                        f"imHeight={info_dict['imHeight']}\n" +
                        f"imExt={info_dict['imExt']}\n"
                    )
        else:
            raise ValueError('dataset_name not supported')
        

        for value_dict in seqmap_dict.values():
            with open(value_dict['path'],'w') as f:
                f.writelines(value_dict['items'])

        with open(save_json_path,'w') as json_file:
            json.dump(save_json, json_file,indent=4)
        print(f'json file saved to {save_json_path}')
if __name__ == '__main__':
    main()
