import argparse
import re
import json
import numpy as np

from datasets import Dataset, DatasetDict, load_dataset
from typing import List

DATASET_ROOT = '/caojiangxia/chenyao/distill_attention/Qwen32B_datasets'


class DatasetLoader(object):
    def __init__(self, dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None):
        self.data_root = DATASET_ROOT
        self.dataset_name = dataset_name
        self.source_dataset_name = source_dataset_name
        self.dataset_version = dataset_version
        self.has_valid = has_valid
        self.split_map = split_map

        self.batch_size = batch_size
        self.train_batch_idxs = train_batch_idxs
        self.test_batch_idxs = test_batch_idxs
        self.valid_batch_idxs = valid_batch_idxs
        
        assert self.split_map is not None    


    def to_json(self, datasets):
        for k, v in self.split_map.items():
            datasets[v].to_json(f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_{k}.json')


    def load_from_json(self): #训练使用的函数,加载原始训练集,返回值dataset是可迭代的对象，有train和test两个键
        data_files = {
            'train': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_train.json',
            'test': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_test.json',
        }

        if self.has_valid:
            data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_valid.json',})

        datasets = load_dataset('json', data_files=data_files)
        datasets = self._post_process(datasets) 
        # print(type(datasets['train']))
        # subsample training dataset if needed
        num_train = len(datasets['train'])
        idxs = list()
        for idx in self.train_batch_idxs:
            idxs += range(idx*self.batch_size, (idx+1)*self.batch_size)        
        datasets['train'] = Dataset.from_dict(datasets['train'][[idx for idx in idxs if idx < num_train]])

        return datasets


    def load_llm_preds(self, split): #训练使用的函数，split的取值是"test" "train"，加载教师模型的rationale,label列表
        labels = list()
        rationales = list()
        for idx in getattr(self, f'{split}_batch_idxs'):
            with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                outputs = json.load(f)

            for output in outputs:
                rationale, label = self._parse_llm_output(output)

                rationales.append(rationale)
                labels.append(label)

        return rationales, labels

   
    def load_teacher_atten(self, t_layer=15)->List[List[List[int]]]:  #训练使用的函数，加载教师模型的注意力矩阵到dataset
        t_atten = list()
        file_path = f'{self.data_root}/{self.dataset_name}/teacher_attention_npy/Qwen2.5-32B/Qwen2.5_layer{t_layer}_attention.npy'
        t_attentions = np.load(file_path, allow_pickle=True)
        for item_atten in t_attentions:
            item_atten = item_atten.tolist()
            t_atten.append(item_atten)
        return t_atten
    
    def load_teacher_allLayer_atten(self)->List[List[List[List[int]]]]:  #训练使用的函数，加载教师模型的所有层的注意力矩阵到dataset
        all_layer_t_atten = list()
        for t_layer in range(1,33):
            layer_t_atten  = []
            file_path = f'{self.data_root}/{self.dataset_name}/teacher_attention_npy/Qwen2.5-32B/Qwen2.5_layer{t_layer}_attention.npy'
            t_attentions = np.load(file_path, allow_pickle=True)
            for item_atten in t_attentions:
                item_atten = item_atten.tolist()
                layer_t_atten.append(item_atten)
            all_layer_t_atten.append(layer_t_atten)
        
        all_sample_t_atten = list()
        sample_num = len(all_layer_t_atten[0])
        for i in range(sample_num):
            sample_t_atten = [layer_atten[i] for layer_atten in all_layer_t_atten]
            all_sample_t_atten.append(sample_t_atten)
        
        return all_sample_t_atten
    

    def _post_process(self, datasets):
        raise NotImplementedError


    def _parse_llm_output(self, output):
        raise NotImplementedError


class SVAMP_Palm_DatasetLoader(DatasetLoader):
    def __init__(self): 
        dataset_name = 'svamp+PALM'
        source_dataset_name = 'svamp+PALM'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(2)
        test_batch_idxs = range(1)


        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)
        

    def _post_process(self, datasets):
        return datasets


    def _parse_llm_output(self, output): 
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' ' #不存在the answer is 就令rationale label 都为空格
            label = ' '
            return rationale, label
        
            
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label

class ASDivDatasetLoader(DatasetLoader):
    def __init__(self): 
        dataset_name = 'asdiv'
        source_dataset_name = 'asdiv'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 900
        train_batch_idxs = range(2)
        test_batch_idxs = range(1)


        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)

    def _post_process(self, datasets):
        return datasets


    def _parse_llm_output(self, output): 
        try:
            rationale, label = output.split('The answer is')
        except:
            rationale = ' ' #不存在the answer is 就令rationale label 都为空格
            label = ' '
            return rationale, label
        
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label
    
    def load_llm_preds(self, split): #训练使用的函数，split的取值是"test" "train"，加载教师模型的rationale,label列表
        labels = list()
        rationales = list()
       
        with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_0.json') as f:
            outputs = json.load(f)

        for output in outputs:
            rationale, label = self._parse_llm_output(output)

            rationales.append(rationale)
            labels.append(label)

        return rationales, labels
    

class SVAMPDatasetLoader(DatasetLoader):
    def __init__(self): 
        dataset_name = 'svamp'
        source_dataset_name = 'svamp'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 700
        train_batch_idxs = range(2)
        test_batch_idxs = range(1)


        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)

    def _post_process(self, datasets):
        return datasets


    def _parse_llm_output(self, output): 
        try:
            rationale, label = output.split('The answer is')
        except:
            rationale = ' ' #不存在the answer is 就令rationale label 都为空格
            label = ' '
            return rationale, label
        
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label
    
    def load_llm_preds(self, split): #训练使用的函数，split的取值是"test" "train"，加载教师模型的rationale,label列表
        labels = list()
        rationales = list()
       
        with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_0.json') as f:
            outputs = json.load(f)

        for output in outputs:
            rationale, label = self._parse_llm_output(output)

            rationales.append(rationale)
            labels.append(label)

        return rationales, labels
    
class MultiArithDatasetLoader(DatasetLoader):
    def __init__(self): 
        dataset_name = 'multiarith'
        source_dataset_name = 'multiarith'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 700
        train_batch_idxs = range(2)
        test_batch_idxs = range(1)


        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)

    def _post_process(self, datasets):
        return datasets


    def _parse_llm_output(self, output): 
        try:
            rationale, label = output.split('The answer is')
        except:
            rationale = ' ' #不存在the answer is 就令rationale label 都为空格
            label = ' '
            return rationale, label
        
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label
    
    def load_llm_preds(self, split): #训练使用的函数，split的取值是"test" "train"，加载教师模型的rationale,label列表
        labels = list()
        rationales = list()
       
        with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_0.json') as f:
            outputs = json.load(f)

        for output in outputs:
            rationale, label = self._parse_llm_output(output)

            rationales.append(rationale)
            labels.append(label)

        return rationales, labels

class SingleEqDatasetLoader(DatasetLoader):
    def __init__(self): 
        dataset_name = 'singleeq'
        source_dataset_name = 'singleeq'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 700
        train_batch_idxs = range(2)
        test_batch_idxs = range(1)


        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)

    def _post_process(self, datasets):
        return datasets


    def _parse_llm_output(self, output): 
        try:
            rationale, label = output.split('The answer is')
        except:
            rationale = ' ' #不存在the answer is 就令rationale label 都为空格
            label = ' '
            return rationale, label
        
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label
    
    def load_llm_preds(self, split): #训练使用的函数，split的取值是"test" "train"，加载教师模型的rationale,label列表
        labels = list()
        rationales = list()
       
        with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_0.json') as f:
            outputs = json.load(f)

        for output in outputs:
            rationale, label = self._parse_llm_output(output)

            rationales.append(rationale)
            labels.append(label)

        return rationales, labels

class GSM8kDatasetLoader(DatasetLoader):
    def __init__(self): 
        dataset_name = 'gsm8k'
        source_dataset_name = 'gsm8k'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 1000
        train_batch_idxs = range(6)
        test_batch_idxs = range(2)


        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)

    def _post_process(self, datasets):
        return datasets


    def _parse_llm_output(self, output): 
        try:
            rationale, label = output.split('The answer is')
        except:
            rationale = ' ' #不存在the answer is 就令rationale label 都为空格
            label = ' '
            return rationale, label
        
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label
    
    def load_llm_preds(self, split): #训练使用的函数，split的取值是"test" "train"，加载教师模型的rationale,label列表
        labels = list()
        rationales = list()
       
        with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_0.json') as f:
            outputs = json.load(f)

        for output in outputs:
            rationale, label = self._parse_llm_output(output)

            rationales.append(rationale)
            labels.append(label)

        return rationales, labels

class MathQADatasetLoader(DatasetLoader):
    def __init__(self): 
        dataset_name = 'mathqa'
        source_dataset_name = 'mathqa'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 800
        train_batch_idxs = range(4)
        test_batch_idxs = range(1)


        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)

    def _post_process(self, datasets):
        return datasets


    def _parse_llm_output(self, output): 
        try:
            rationale, label = output.split('The answer is')
        except:
            rationale = ' ' #不存在the answer is 就令rationale label 都为空格
            label = ' '
            return rationale, label
        
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label
    
    def load_llm_preds(self, split): #训练使用的函数，split的取值是"test" "train"，加载教师模型的rationale,label列表
        labels = list()
        rationales = list()
       
        with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_0.json') as f:
            outputs = json.load(f)

        for output in outputs:
            rationale, label = self._parse_llm_output(output)

            rationales.append(rationale)
            labels.append(label)

        return rationales, labels


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, required=True)
    # args = parser.parse_args()
    dataset_loader = MathQADatasetLoader()
    datasets = dataset_loader.load_from_json()
    train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split="train")
    test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds(split="test")
    datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
    datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
    datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
    datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)
    t_attentions = dataset_loader.load_teacher_allLayer_atten()
    datasets['train'] = datasets['train'].add_column('t_atten', t_attentions)
  




    

 

