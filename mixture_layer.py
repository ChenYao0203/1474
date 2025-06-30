import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import  GPT2LMHeadModel, PreTrainedModel, GPT2Config
from transformers import  AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from easydict import EasyDict


class RMSNorm(nn.Module):
  def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None: #hidden_dim是所有的头的V值拼接在一起的维度
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(hidden_dim))
  
  def _norm(self, v_tensor: Tensor) -> Tensor: # v_tensor的shape是Batch_size X 1 X d
    variance = v_tensor.pow(2).mean(-1, keepdim=True)
    return v_tensor * torch.rsqrt(variance + self.eps)
  
  def forward(self, v_tensor: Tensor) -> Tensor:
    return self.weight * self._norm(v_tensor).type_as(v_tensor)


class S_MixOfLayer(nn.Module):
    def __init__(self, hidden_dim:int, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.rms_norm = RMSNorm(hidden_dim = hidden_dim)
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, v_tensor_groups, attention_mask): 
        #v_tensor_groups 元组或列表 L个tensor: Batch_size X Max_len X d 
        # attention_mask: Batch_size X Max_len
        v_tensors = torch.stack(v_tensor_groups, dim=1) #v_tensor: Batch_size x L x Max_len x d 
        v_tensors = v_tensors * attention_mask.unsqueeze(-1).unsqueeze(1) #v_tensor: Batch_size x L x Max_len x d 
        v_tensors = v_tensors.sum(-2) #v_tensor: Batch_size x L x d 
        v_tensors = self.rms_norm(v_tensors)  #v_tensor: Batch_size x L x d 
        L_weight = self.gate(v_tensors) #L_weight: Batch_size x L x 1 
        L_weight = L_weight.squeeze() #L_weight: Batch_size x L
        # L_weight = L_weight.sum(0) #L_weight: L
        L_weight = L_weight / (self.temperature + 1e-6)
        L_weight = F.softmax(L_weight, dim=-1)  
        return L_weight #L_weight: Batch_size x L

#获取教师模型权重
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


#定义钩子函数 TinyLlama
class QKVExtractor:
    def __init__(self):
        self.v_values = []

    def student_hook_fn(self, module, input, output):
        # output 是经过v_proj后的value_states
        # [batch_size, seq_len, hidden_size]
        # print(output.shape)

        self.v_values.append(output.detach())
        
    def teacher_hook_fn(self, module, input, output):
        # 教师模型的 hook 逻辑（保持不变）
        if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight'):
            self.v_values.append(output.detach())


class StudentModel(nn.Module):
    def __init__(self, student_model_path, temperature=1.0):
        super().__init__()
        
        # 初始化学生模型
        self.student_model = AutoModelForCausalLM.from_pretrained(student_model_path, torch_dtype=torch.bfloat16, attn_implementation="eager")
        self.generation_config = self.student_model.generation_config  # 继承生成配置
        
        # mix_layer初始化
        s_hidden_dim = int(self.student_model.config.hidden_size / self.student_model.config.num_attention_heads * self.student_model.config.num_key_value_heads)
        self.student_mix_layer = S_MixOfLayer(s_hidden_dim, temperature)
       
        # QKV提取器实例化
        self.stu_qkv_extractor = QKVExtractor()
        self.hook_handles = []  # 存储 hook 句柄

    def forward(self, **inupts):
        return self.student_model(**inupts)
    
    def generate(self, input_ids, attention_mask, **gen_kwargs):
        return self.student_model.generate(input_ids, attention_mask=attention_mask,**gen_kwargs)

    def rationale_forward(self, output_attentions=True, **input):
            self.hook_handles = []
            # 学生模型注册钩子
            for layer in self.student_model.model.layers:
                handle = layer.self_attn.v_proj.register_forward_hook(self.stu_qkv_extractor.student_hook_fn)
                self.hook_handles.append(handle)

            self.stu_qkv_extractor.v_values = [] #清空批次
            
            student_output = self.student_model(**input, output_attentions=output_attentions)

            #获取学生权重
            s_v_tensors =  self.stu_qkv_extractor.v_values #执行过forward函数之后再执行该函数
            stu_L_weight = self.student_mix_layer(s_v_tensors, input['attention_mask'])
            
            
            # 执行完前向传播后注销钩子
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles = []

            return EasyDict({"loss": student_output.loss, "s_attentions":student_output.attentions, "s_L_weight":stu_L_weight})
    
if __name__ == '__main__':
    model  = StudentModel(student_model_path="/root/autodl-tmp/model_base/gpt2_large", teacher_model_path="/root/autodl-tmp/model_base/LLAMA3_CHIN")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

