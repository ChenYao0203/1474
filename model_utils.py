import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch.nn.functional as F
from attention_loss import number_index, atten_loss_kl, mol_atten_loss_kl
from mixture_layer import get_t_layer_weight
import os
import logging

TRAINING_ARGS_NAME = "training_args.bin"
class StandardDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors="pt"): #features是一个列表，每个元素是字典，字典中包含了input_ids和label_ids
        features_df = pd.DataFrame(features)
        #三种key 在run.py中设置, 一个任务
        pred_features = features_df.loc[:, features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].to_dict('records')
        pred_features = super().__call__(pred_features, return_tensors) #父类方法按批次将列表数据转化为Tensor
        return {
            'pred': pred_features
        }
    
class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors="pt"): #features是一个列表，每个元素是字典，字典中包含了input_ids和label_ids
       
        features_df = pd.DataFrame(features)
        #六种key 在run.py中设置, 两个任务，pred是label，expl是rationale
        pred_features = features_df.loc[:, features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].to_dict('records')
        expl_features = features_df.loc[:, features_df.columns.isin(['aux_labels', 'expl_input_ids', 'expl_attention_mask'])].rename(
            columns={'aux_labels': 'labels', 'expl_input_ids': 'input_ids', 'expl_attention_mask': 'attention_mask'}).to_dict('records')
        pred_features = super().__call__(pred_features, return_tensors) #父类方法按批次将列表数据转化为Tensor
        
        #mol处理
        # if "t_input_ids" in features[0]:
        #     t_atten = features_df.loc[:, features_df.columns.isin(['t_atten_alllayer'])].to_dict('records')
        #     t_atten = [torch.tensor(sample['t_atten_alllayer']) for sample in t_atten]
            
        #     expl_features_input_ids = [sample['input_ids'] for sample in expl_features]
        #     max_length = max([len(ii) for ii in expl_features_input_ids])
        #     dist = [max_length-len(ii) for ii in expl_features_input_ids] 
        #     s_atten_ddindex = features_df.loc[:, features_df.columns.isin(['s_atten_ddindex'])].to_dict('records')
        #     s_atten_ddindex = [(sample['s_atten_ddindex'][0],sample['s_atten_ddindex'][1],d) for sample,d in zip(s_atten_ddindex,dist)] 
            
        #     expl_features = super().__call__(expl_features, return_tensors)

        #     teacher_features = features_df.loc[:, features_df.columns.isin(['t_input_ids', 't_attention_mask'])].rename(
        #     columns={'t_input_ids': 'input_ids', 't_attention_mask': 'attention_mask'}).to_dict('records')
        #     teacher_features = super().__call__(teacher_features, return_tensors)

        #     return {
        #         'pred': pred_features,
        #         'expl': expl_features,
        #         't_inputs': teacher_features,
        #         't_attenions':t_atten,
        #         's_atten_ddindex': s_atten_ddindex
        #     }

        #对于教师注意力进行额外的处理，并获取padding之后的dd_index的dist
        if "t_atten" in features[0]:
            t_atten = features_df.loc[:, features_df.columns.isin(['t_atten'])].to_dict('records')
            t_atten = [torch.tensor(sample['t_atten']) for sample in t_atten]

            expl_features_input_ids = [sample['input_ids'] for sample in expl_features]
            max_length = max([len(ii) for ii in expl_features_input_ids])
            dist = [max_length-len(ii) for ii in expl_features_input_ids] 
            s_atten_ddindex = features_df.loc[:, features_df.columns.isin(['s_atten_ddindex'])].to_dict('records')
            s_atten_ddindex = [(sample['s_atten_ddindex'][0],sample['s_atten_ddindex'][1],d) for sample,d in zip(s_atten_ddindex,dist)] 
            
            expl_features = super().__call__(expl_features, return_tensors)
            return {
                'pred': pred_features,
                'expl': expl_features,
                't_atten':t_atten,
                's_atten_ddindex': s_atten_ddindex
            }
        
        else:
            expl_features = super().__call__(expl_features, return_tensors)
            # print(f"Pred features: {len(pred_features)}, Expl features: {len(expl_features)}")
            return {
                'pred': pred_features,
                'expl': expl_features,
            }

#重写prediction_step函数，用于仅解码器架构的预测 
class Decoder_StandardTrainer(Seq2SeqTrainer):
    def __init__(self, callbacks=None, **kwargs): 
        super().__init__(callbacks=callbacks,**kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        pred_outputs = model(**inputs['pred'])
        loss = pred_outputs.loss
        return loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        torch.cuda.empty_cache()
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        pred_input_lenth = inputs['pred']['input_ids'].shape[-1]
        pred_outputs = list(pred_outputs) 
        pred_outputs[1] = pred_outputs[1][:,pred_input_lenth:]
        
        return (
            pred_outputs[0], #loss是None
            pred_outputs[1], #模型预测的token_id
            pred_outputs[2], #labels
        )


#重写prediction_step函数，用于仅解码器架构的预测
class Decoder_TaskPrefixTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, output_rationale, beta, attention_distill, s_atten_layer, kl_temperature, MOL_attention_distill, t_temperature, MMI_distill, layer_weight_path, callbacks=None, **kwargs): 
        super().__init__(callbacks=callbacks,**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale
        self.attention_distill = attention_distill
        self.beta = beta
        self.s_atten_layer = s_atten_layer
        self.kl_temperature = kl_temperature
        self.MMI_distill = MMI_distill
        self.MOL_attention_distill = MOL_attention_distill
        self.layer_weight_path = layer_weight_path
        self.t_temperature = t_temperature
        # self.init_teacher_weight = torch.zeros(1)
        # self.init_student_weight = torch.zeros(1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        #这里应该输出学生模型的注意力，然后和教师模型计算注意力loss
        pred_outputs = model(**inputs['pred'])
        loss = self.alpha * pred_outputs.loss
        
        #固定 映射学生和教师的注意力差距
        if self.attention_distill:
            # if self.kl_loss_only:
            expl_outputs = model(**inputs['expl'], output_attentions=True)
            atten_loss = atten_loss_kl(expl_outputs.attentions, inputs['t_atten'], inputs['s_atten_ddindex'], self.s_atten_layer,self.kl_temperature)
            loss = loss + (1. - self.alpha) * expl_outputs.loss + self.beta * atten_loss

        #动态 映射学生和教师的注意力差距
        elif self.MOL_attention_distill:
            expl_outputs =  model.rationale_forward(output_attentions=True, **inputs['expl'])
            t_L_weight =  get_t_layer_weight(inputs['t_atten'], self.t_temperature)
            # if self.state.global_step == 0:
            #     self.init_teacher_weight = t_L_weight.detach().clone()
            #     self.init_student_weight = expl_outputs.s_L_weight.detach().clone()
            #     self.init_teacher_weight = self.init_teacher_weight.mean(dim=0, keepdim=True)
            #     self.init_student_weight = self.init_student_weight.mean(dim=0, keepdim=True)
            mol_atten_loss = mol_atten_loss_kl(expl_outputs.s_attentions, expl_outputs.s_L_weight, inputs['t_atten'], t_L_weight, inputs['s_atten_ddindex'], self.kl_temperature)
            loss = loss + (1. - self.alpha) * expl_outputs.loss + self.beta * mol_atten_loss
            if self.layer_weight_path:
                with open(self.layer_weight_path, 'a', encoding='utf-8') as f:
                    s_L_weight_list = expl_outputs.s_L_weight.tolist()
                    t_L_weight_list = t_L_weight.tolist()
                    log_entry = {
                        "s_L_weight": s_L_weight_list,
                        "t_L_weight": t_L_weight_list
                    }
                    f.write(json.dumps(log_entry) + "\n")
           
                

        #MMI 对比方法
        elif self.MMI_distill:
            expl_outputs = model(**inputs['expl'])
            pred_logits = (pred_outputs.logits).detach()
            expl_logits = (expl_outputs.logits)
            pred_prob = pred_logits.softmax(dim=-1)
            expl_prob = expl_logits.softmax(dim=-1)
            
            pred_prob_1 = torch.max(pred_prob, dim=-2)[0]
            expl_prob_1 = torch.max(expl_prob, dim=-2)[0]
            
            Loss = nn.CrossEntropyLoss()
            ms_loss = Loss(expl_prob_1,pred_prob_1)
            loss = loss + (1. - self.alpha) * expl_outputs.loss + self.beta* ms_loss
        
        #对比方法
        else:
            expl_outputs = model(**inputs['expl'])
            loss = loss + (1. - self.alpha) * expl_outputs.loss #loss函数的设计
        
        print(f"loss: {loss.item():.4f}")  # 转为 float 并保留4位小数
        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        torch.cuda.empty_cache()
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        
       
        if self.output_rationale:
            expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)
        else:
            expl_outputs = pred_outputs # placeholder only
        
        # print("********************")
        # print(type(inputs['pred']['input_ids']))
        # print(inputs['pred']['input_ids'].shape)
        # print(type(pred_outputs[1]))
        # print(type(expl_outputs[1]))
        # print(pred_outputs[1].size())
        # print(expl_outputs[1].size())
        
        pred_input_lenth = inputs['pred']['input_ids'].shape[-1]
        pred_outputs = list(pred_outputs) 
        pred_outputs[1] = pred_outputs[1][:,pred_input_lenth:]
        
        expl_input_lenth = inputs['expl']['input_ids'].shape[-1]
        expl_outputs = list(expl_outputs) 
        expl_outputs[1] = expl_outputs[1][:,expl_input_lenth:]

        # print("********************")
        # print(type(inputs['pred']['input_ids']))
        # print(inputs['pred']['input_ids'].shape)
        # print(type(pred_outputs[1]))
        # print(pred_outputs[1].size())
        # print(type(expl_outputs[1]))
        # print(expl_outputs[1].size())

        # print(pred_outputs[1][0])
        # print(pred_outputs[1][1])
        #修改了seq2seq_trainer的predict_step函数，仅解码架构不需要在测试时计算loss
        # loss = self.alpha * pred_outputs[0]  + (1 - self.alpha) * expl_outputs[0]
        # logging.info(f"Prediction step loss: {loss.item()}")
        
        return (
            pred_outputs[0], #loss是None
            [pred_outputs[1], expl_outputs[1]], #模型预测的token_id
            [pred_outputs[2], expl_outputs[2]], #labels
        )
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.MOL_attention_distill:
            self.model.student_model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        
            

#测试datacollator
if __name__ == '__main__':
    from datasets import DatasetDict
    from data_utils import  SVAMPDatasetLoader
    dataset_loader = SVAMPDatasetLoader()
    datasets = dataset_loader.load_from_json()
    train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split='train')
    test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds(split='test')
    datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
    datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
    datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
    datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)
    
    datasets = DatasetDict({
            'train': datasets['train'],
            'test': datasets['test'],
        })
    if 'rationale' in datasets['train'].column_names:
            datasets = datasets.remove_columns('rationale')
    datasets = datasets.rename_column('llm_rationale', 'rationale')
    
    #args.attention_distill为true时，train_dataset中添加教师注意力
    # t_attentions = dataset_loader.load_teacher_atten()
    # datasets['train'] = datasets['train'].add_column('t_atten', t_attentions)
    t_attentions = dataset_loader.load_teacher_allLayer_atten()
    datasets['train'] = datasets['train'].add_column('t_atten', t_attentions)
    
    model_path = "/root/autodl-tmp/model_base/gpt2_large"
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_train_function(examples): #训练集
        eos_token_id = tokenizer.eos_token_id #手动给训练的输入和标签的末尾加入eos_token_id
        model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=1024, truncation=True)
        expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=1024, truncation=True)
        model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
        
        label_output_encodings = tokenizer(examples['label'], max_length=512, truncation=True) #设置标签的最大生成长度为256
        rationale_output_encodings = tokenizer(examples['rationale'], max_length=512, truncation=True)
        
        model_inputs['labels'] =  [[-100] * len(iid) + lid + [eos_token_id]  for iid,lid in zip(model_inputs["input_ids"], label_output_encodings['input_ids'])]
        model_inputs["input_ids"] = [iid+lid+ [eos_token_id] for iid,lid in zip(model_inputs["input_ids"], label_output_encodings['input_ids'])]
        model_inputs['attention_mask'] =[[1] * len(iid) for iid in model_inputs["input_ids"]]
        
        model_inputs['aux_labels']  = [[-100] * len(iid) + lid +  [eos_token_id]  for iid,lid in zip(model_inputs["expl_input_ids"], rationale_output_encodings['input_ids'])]
        model_inputs['expl_input_ids'] = [iid+lid+ [eos_token_id] for iid,lid in zip(model_inputs["expl_input_ids"], rationale_output_encodings['input_ids'])]
        model_inputs['expl_attention_mask'] = [[1] * len(iid) for iid in model_inputs["expl_input_ids"]]
        
        ##args.attention_distill为true时，学生模型要在训练集中存储对于query+COT的digit_index和dot_index
        decoded_tokens = [tokenizer.convert_ids_to_tokens(sample_expl_input_id) for sample_expl_input_id in model_inputs['expl_input_ids']]
        model_inputs['s_atten_ddindex'] = [number_index(sample_decoded_tokens) for sample_decoded_tokens in decoded_tokens]
        
        return model_inputs

    def tokenize_test_function(examples): #测试集
        model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=1024, truncation=True)
        expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=1024, truncation=True)
        model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
        model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
        with tokenizer.as_target_tokenizer():
            label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True) #设置标签的最大生成长度为256
            rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)
        model_inputs['labels'] = label_output_encodings['input_ids']
        model_inputs['aux_labels'] = rationale_output_encodings['input_ids']
       
        return model_inputs 
    
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
            'test': tokenized_test_datasets
            })
    
    data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    train_batch = data_collator(tokenized_datasets["train"])
    test_batch = data_collator(tokenized_datasets["test"])

    batch = train_batch['t_atten'][:16]


    def get_t_layer_weight(teacher_attentions, temperature=0.1):
        temp = [] 
        for sample_attention in teacher_attentions:
            gradients = torch.abs(sample_attention[:, :, 1:] - sample_attention[:, :, :-1])  # L x step x digit-1
            sample_mean = gradients.sum(dim=-1).sum(dim=-1)  # L
            temp.append(sample_mean)
        L_weight = torch.stack(temp,dim=0) # B x L 
        L_weight = L_weight / (temperature + 1e-6) 
        L_weight = F.softmax(L_weight, dim=-1) # B x L 
        return L_weight #L_weight: Batch_size x L
        
    for i in get_t_layer_weight(batch):
        print(i)

    


    
    


       
  
