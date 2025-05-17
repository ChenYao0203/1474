import argparse

from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

from data_utils import SVAMPDatasetLoader, ASDivDatasetLoader, MultiArithDatasetLoader, SingleEqDatasetLoader, SVAMP_Palm_DatasetLoader, GSM8kDatasetLoader, MathQADatasetLoader
from metrics import compute_text_acc, compute_equation_acc, compute_metrics_text, compute_metrics_equation, compute_metrics_text_aux, compute_metrics_equation_aux
from train_utils import train_and_evaluate
import numpy as np
from attention_loss import number_index
import torch

def run(args):
    #### Prepare datasets
    if args.dataset == 'multiarith':
        dataset_loader = MultiArithDatasetLoader()
    elif args.dataset == 'svamp':
        dataset_loader = SVAMPDatasetLoader()
    elif args.dataset == 'svamp+PALM':
        dataset_loader = SVAMP_Palm_DatasetLoader()
    elif args.dataset == 'singleeq':
        dataset_loader = SingleEqDatasetLoader()
    elif args.dataset == 'asdiv':  
        dataset_loader = ASDivDatasetLoader()
    elif args.dataset == 'gsm8k':  
        dataset_loader = GSM8kDatasetLoader()
    elif args.dataset == 'mathqa':  
        dataset_loader = MathQADatasetLoader()
    else:
        raise ValueError

  
    datasets = dataset_loader.load_from_json()

  
    #是否使用教师模型的COT数据
    if args.llm == 'no_llm':  
        pass
    elif args.llm == 'palm': 
        train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split='train') #COT0和COT1 都包括了
        test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds(split='test')
        datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
        datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
        datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
        datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)
    else:
            raise ValueError
       

    #是否加载注意力
    if args.dataset == 'svamp' or args.dataset == "gsm8k":
        if args.attention_distill:  #args.attention_distill为true时，train_dataset中添加教师注意力矩阵
            t_attentions = dataset_loader.load_teacher_atten(t_layer=args.t_atten_layer)
            datasets['train'] = datasets['train'].add_column('t_atten', t_attentions)
        elif args.MOL_attention_distill:  #args.MOL_attention_distill为true时，train_dataset中添加所有层的教师注意力矩阵
            t_attentions = dataset_loader.load_teacher_allLayer_atten()
            datasets['train'] = datasets['train'].add_column('t_atten', t_attentions)
        else:
            pass
    
    #是否对训练集进行子采样
    if args.subsample < 1.0: 
        # datasets['train'] = datasets['train'].train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']
        datasets['test'] = datasets['test'].train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']

    #使用groundtruth或者教师模型生成的结果 作为结果label
    if args.label_type == 'gt': # ground truth，此时的dataset包含四个键，input问题,label GT的标签, llm_label 教师模型的标签, llm_rationale 教师模型的rationale
        pass
    elif args.label_type == 'llm' and args.llm == "palm": #llm 将GT的label换成教师模型生成的label, input问题,label 教师模型的标签, llm_label 教师模型的标签, llm_rationale 教师模型的rationale
        datasets['train'] = datasets['train'].remove_columns('label')
        datasets['train'] = datasets['train'].add_column('label', datasets['train']['llm_label'])

    else:
        raise ValueError

    if args.llm == "palm":
        if 'rationale' in datasets['train'].column_names:
            datasets = datasets.remove_columns('rationale')
        datasets = datasets.rename_column('llm_rationale', 'rationale')


    # decoder-only结构手动设置padding_side为left和pad_token
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # # mol方法加载教师模型的tokenizer
    # if args.MOL_attention_distill:
    #     teacher_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/model_base/LLAMA3-8B", padding_side='left')
    #     teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    if args.task_type == 'task_prefix' and args.llm == "palm": #多任务训练
        #decoder-only的训练集和测试集分开进行tokenizer, model_inputs['expl_input_ids']是一个列表，其他同理
        def tokenize_train_function(examples): #训练集
            eos_token_id = tokenizer.eos_token_id #手动给训练的输入和标签的末尾加入eos_token_id
            model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
    
            
            label_output_encodings = tokenizer(examples['label'], max_length=512, truncation=True) #设置标签的最大生成长度为256
            rationale_output_encodings = tokenizer(examples['rationale'], max_length=512, truncation=True)
            
            model_inputs['labels'] =  [[-100] * len(iid) + lid + [eos_token_id]  for iid,lid in zip(model_inputs["input_ids"], label_output_encodings['input_ids'])]
            model_inputs["input_ids"] = [iid+lid+ [eos_token_id] for iid,lid in zip(model_inputs["input_ids"], label_output_encodings['input_ids'])]
            model_inputs['attention_mask'] =[[1] * len(iid) for iid in model_inputs["input_ids"]]
            
            model_inputs['aux_labels']  = [[-100] * len(iid) + lid +  [eos_token_id]  for iid,lid in zip(model_inputs["expl_input_ids"], rationale_output_encodings['input_ids'])]
            model_inputs['expl_input_ids'] = [iid+lid+ [eos_token_id] for iid,lid in zip(model_inputs["expl_input_ids"], rationale_output_encodings['input_ids'])]
            model_inputs['expl_attention_mask'] = [[1] * len(iid) for iid in model_inputs["expl_input_ids"]]
            
            #attention_distill或者MOL_attention_distill都需要获取 s_atten_ddindex：List[List[digit_index,dot_index]]
            if args.dataset == 'svamp' or args.dataset == "gsm8k":
                if  args.attention_distill or args.MOL_attention_distill:
                    decoded_tokens = [tokenizer.convert_ids_to_tokens(sample_expl_input_id) for sample_expl_input_id in model_inputs['expl_input_ids']]
                    model_inputs['s_atten_ddindex'] = [number_index(sample_decoded_tokens) for sample_decoded_tokens in decoded_tokens] #digit_index还没有合并
                # for sample_index,sample_tokens in zip(model_inputs['s_atten_ddindex'],decoded_tokens):
                #     if not sample_index[0]:
                #         print("抽取的digit_index为空：",sample_index[0]) #测试学生模型的抽取结果
                #         print("对应的decoded_tokens为：",sample_tokens)
            
            #MOL_attention_distill需要获取教师模型的input_ids和attetion_mask
            # if args.MOL_attention_distill:
            #     teacher_tokenized = teacher_tokenizer(['explain: ' + text for text in examples['input']], max_length=1024, truncation=True)
            #     teacher_rationale_output_encodings = teacher_tokenizer(examples['rationale'], max_length=512, truncation=True)

            #     teacher_input_ids = teacher_tokenized["input_ids"]
            #     teacher_rationale_output_encodings = teacher_rationale_output_encodings['input_ids']
            
            #     teacher_eos_token_id = teacher_tokenizer.eos_token_id
            #     model_inputs['t_input_ids'] = [iid+lid+[teacher_eos_token_id] for iid,lid in zip(teacher_input_ids, teacher_rationale_output_encodings)]
            #     model_inputs['t_attention_mask'] = [[1] * len(iid) for iid in model_inputs['t_input_ids']]

            return model_inputs
        
        def tokenize_test_function(examples): #测试集
            model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True) #设置标签的最大生成长度为256
                rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)
            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']
            return model_inputs 

    elif args.task_type == 'standard': #单任务训练
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples['input'],
                max_length=args.max_input_length,
                truncation=True
            )
            eos_token_id = tokenizer.eos_token_id 
            label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
            label_output_encodings = label_output_encodings['input_ids']
            model_inputs['labels'] =  [[-100] * len(iid) + lid + [eos_token_id]  for iid,lid in zip(model_inputs["input_ids"], label_output_encodings)]
            model_inputs["input_ids"] = [iid+lid+ [eos_token_id] for iid,lid in zip(model_inputs["input_ids"], label_output_encodings)]
            model_inputs['attention_mask'] =[[1] * len(iid) for iid in model_inputs["input_ids"]]
            return model_inputs

    else:
        raise ValueError

    #选择tokneizer_fuinction批次处理dataset
    if args.task_type == 'standard':
        if "answer" in datasets.column_names:
            tokenized_datasets = datasets.map(
                tokenize_function,
                remove_columns=['input', 'rationale', 'label', 'llm_label', 'answer'],
                batched=True
            )
        else:
            tokenized_datasets = datasets.map(
                tokenize_function,
                remove_columns=['input', 'rationale', 'label', 'llm_label'],
                batched=True
            )
    else:
        tokenized_train_datasets = datasets['train'].map(
            tokenize_train_function,
            remove_columns=['input', 'rationale', 'label', 'llm_label'],
            batched=True
        )
        tokenized_test_datasets = datasets["test"].map(
            tokenize_test_function,
            remove_columns=['input', 'rationale', 'label', 'llm_label'],
            batched=True
        )
        tokenized_datasets = DatasetDict({
        'train': tokenized_train_datasets,
        'test': tokenized_test_datasets,
        })

    
    if args.task_type == 'standard':
        compute_metrics = compute_metrics_equation_aux(tokenizer)

    else:
        compute_metrics = compute_metrics_equation(tokenizer)


    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--label_type', type=str, default='gt')
    parser.add_argument('--llm', type=str, default='palm') #palm:加载教师的COT no_llm:不加载教师的COT
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--task_type', type=str, default='task_prefix') #task_prefix:加载教师的COT standard:不加载教师的COT
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    #新增超参数
    parser.add_argument('--attention_distill', action='store_true')
    parser.add_argument('--beta',type=float, default=0.1)
    parser.add_argument('--t_atten_layer', type=int, default=15)
    parser.add_argument('--s_atten_layer', type=int, nargs='+', default=[17,18])
    parser.add_argument('--kl_temperature', type=float, default=0.1)
    parser.add_argument('--MMI_distill', action='store_true')
    parser.add_argument('--MOL_attention_distill', action='store_true')
    parser.add_argument('--mol_temperature', type=float, default=0.1)
    parser.add_argument('--t_temperature', type=float, default=0.1)
    parser.add_argument('--log_weight', action='store_true') #是否存储层权重
    parser.add_argument('--ood_evalu_only', action='store_true') #是否只进行ood测试
   
    
    args = parser.parse_args()

    run(args)