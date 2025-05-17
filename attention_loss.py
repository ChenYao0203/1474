import torch
import torch.nn.functional as F
import numpy as  np

#度量每个时间点的注意分布差异 固定映射
def atten_loss_kl(s_attentions, t_attentions, s_atten_ddindex, s_atten_layer, temperature=1.0):
    
    s_atten = step_attention(s_attentions, s_atten_ddindex, s_atten_layer)
    
    total_kl_div_loss = 0.0 
    batch_size = len(t_attentions)
    for s, t in zip(s_atten, t_attentions):
        if s.shape == t.shape:
            logits = s/(temperature+0.0001)
            target = t/(temperature+0.0001)
            logits = F.log_softmax(logits, dim=-1)  
            target = F.softmax(target, dim=-1)  
            kl_div_loss = F.kl_div(logits, target, reduction='batchmean')  
            total_kl_div_loss += kl_div_loss

    total_kl_div_loss = total_kl_div_loss / batch_size 

    return total_kl_div_loss


#度量每个时间点的注意分布差异 动态映射
def mol_atten_loss_kl(s_attentions, s_L_weight, t_attentions, t_L_weight, s_atten_ddindex, temperature=1.0):
    s_attens = step_weight_attention(s_attentions, s_atten_ddindex)
    
    # #残差连接
    # s_L_weight = s_L_weight * 0.5 +  init_student_weight * 0.5
    # t_L_weight = t_L_weight * 0.5 +  init_teacher_weight * 0.5

    #s_L_weight: batch_size x layer_num s_attens: list(layer_num x step_num x digit_num) len(list) = batch_size 
    #s_atten: step_num x digit_num
    s_atten = [torch.sum(sample_s_weight.unsqueeze(dim=1).unsqueeze(dim=2) * sample_s_atten, dim=0) for sample_s_atten, sample_s_weight in zip(s_attens, s_L_weight)]
    
    #t_L_weight: batch_size x layer_num t_attentions: list(layer_num x step_num x digit_num) len(list) = batch_size
    #t_atten: step_num x digit_num
    t_atten = [torch.sum(sample_t_weight.unsqueeze(dim=1).unsqueeze(dim=2) * sample_t_atten, dim=0) for sample_t_atten, sample_t_weight in zip(t_attentions, t_L_weight)]
    
    total_kl_div_loss = 0.0 
    batch_size = len(t_attentions)
    for s, t in zip(s_atten, t_atten):
        # print(f"student_shape:{s.shape}*****************teacher_shape:{t.shape}")
        if s.shape == t.shape:
            logits = s/(temperature+0.0001)
            target = t/(temperature+0.0001)
            logits = F.log_softmax(logits, dim=-1)  
            target = F.softmax(target, dim=-1)  
            kl_div_loss = F.kl_div(logits, target, reduction='batchmean')  
            total_kl_div_loss += kl_div_loss

    total_kl_div_loss = total_kl_div_loss / batch_size 

    return total_kl_div_loss

#度量整体的注意力分布差异
def atten_loss_pearson(s_attentions, t_attentions, s_atten_ddindex, s_atten_layer, temperature=1.0):
      
    s_atten = step_attention(s_attentions, s_atten_ddindex, s_atten_layer)
    
    total_pearson_loss = 0.0 
    batch_size = len(t_attentions)
    for s, t in zip(s_atten, t_attentions):
        if s.shape == t.shape:
            pearson_loss = 0.0
            num_step = t.shape[0]
            logits = s/(temperature+0.0001)
            target = t/(temperature+0.0001)
            logits = F.softmax(logits, dim=-1)  
            target = F.softmax(target, dim=-1)  
            for x,y in zip(logits.T,target.T):
                pearson_loss += pearson_correlation(x, y,0.0001)
            if num_step!=0:
                pearson_loss += pearson_loss/num_step
            total_pearson_loss += pearson_loss
    
    total_pearson_loss = total_pearson_loss / batch_size 

    return total_pearson_loss




#根据字符位置找到对应的 token 索引
def number_index(decoded_tokens):
    digit_indexes = []
    dot_indexes = []
    flag_q_end = 0 #问题中的句号不算，整个问题算作一个步骤
    len_decoded_tokens = len(decoded_tokens)
    for idx, token in enumerate(decoded_tokens):
        if any(char.isdigit() for char in token):
            if token=="<0x0A>":
                pass
            else:
                digit_indexes.append(idx)
        elif flag_q_end==0 and token in ["?","?Ċ"]:
            dot_indexes.append(idx)
            flag_q_end = 1
        elif flag_q_end==1 and token in [".",".Ċ",".ĊĊ"]: 
            if idx+1 < len_decoded_tokens:
                if decoded_tokens[idx-1]=="▁P" and decoded_tokens[idx+1]=="E":
                    pass
            else:
                dot_indexes.append(idx)
    return digit_indexes,dot_indexes

#将tokenizer中分开进行编码的数字进行合并     
def cat_digit_index(digit_indexes):
    cat_digit_indexes = []
    len_digit_indexes = len(digit_indexes)
    index = 0
    while index < len_digit_indexes:
        temp_digit_index = []
        while index < len_digit_indexes - 1 and digit_indexes[index + 1] == digit_indexes[index] + 1:
            temp_digit_index.append(digit_indexes[index])
            index += 1
        temp_digit_index.append(digit_indexes[index])
        cat_digit_indexes.append(temp_digit_index)
        index += 1
    return cat_digit_indexes


#取学生的逐步注意力，比并对数字进行合并
def step_attention(s_attentions, s_atten_ddindex, s_atten_layer):
        seq_len = s_attentions[0].shape[-1]
        mean_head_attentions = torch.stack([s_attentions[layer_index-1] for layer_index in s_atten_layer]).mean(dim=0) #求层平均值
        mean_head_attentions = mean_head_attentions.mean(dim=1)

        s_step_attentions = []
        for sample_attention, sample_ddindex in zip(mean_head_attentions,s_atten_ddindex):
            sample_step_attentions = []
            start = 0
            dot_indexes = sample_ddindex[1]
            digit_indexes = cat_digit_index(sample_ddindex[0])
            dist = sample_ddindex[2]
            sample_attention = sample_attention[dist:,dist:]
            for index in dot_indexes:
                step_attention = sample_attention[start:index+1,:].sum(dim=0)
                sample_step_attentions.append(step_attention)
                start = index + 1
            if start < seq_len:
                sample_step_attentions.append(sample_attention[start:,:].sum(dim=0))
            sample_step_attentions = torch.stack(sample_step_attentions, dim=0)
            try:
                sample_step_attentions = torch.cat([sample_step_attentions[:,digit_index].sum(dim=-1,keepdim=True) for digit_index in digit_indexes], dim=-1)
            except:
                print("错误样例的digit_indexes",digit_indexes)
                print("错误样例的sample_ddindex", sample_ddindex[0])

            s_step_attentions.append(sample_step_attentions)
        # print(all_sample_step_attentions[-1].shape)
        return s_step_attentions

# 取学生的逐步注意力，比并对数字进行合并，所有层进行加权
def step_weight_attention(s_attentions, s_atten_ddindex):
        seq_len = s_attentions[0].shape[-1]
        batch_num = s_attentions[0].shape[0]
        
        all_layer_attentions = []
        for mean_head_attentions in s_attentions:
            mean_head_attentions = mean_head_attentions.mean(dim=1)

            s_step_attentions = []
            for sample_attention, sample_ddindex in zip(mean_head_attentions,s_atten_ddindex):
                sample_step_attentions = []
                start = 0
                dot_indexes = sample_ddindex[1]
                digit_indexes = cat_digit_index(sample_ddindex[0])
                dist = sample_ddindex[2]
                sample_attention = sample_attention[dist:,dist:]
                for index in dot_indexes:
                    step_attention = sample_attention[start:index+1,:].sum(dim=0)
                    sample_step_attentions.append(step_attention)
                    start = index + 1
                if start < seq_len:
                    sample_step_attentions.append(sample_attention[start:,:].sum(dim=0))
                sample_step_attentions = torch.stack(sample_step_attentions, dim=0)
                try:
                    sample_step_attentions = torch.cat([sample_step_attentions[:,digit_index].sum(dim=-1,keepdim=True) for digit_index in digit_indexes], dim=-1)
                except:
                    # print("错误样例的digit_indexes",digit_indexes)
                    # print("错误样例的sample_ddindex", sample_ddindex[0])
                    pass

                s_step_attentions.append(sample_step_attentions)
            all_layer_attentions.append(s_step_attentions)
        
        all_sample_layer_attentions = []
        for sample_index in range(batch_num):
            all_sample_layer_attentions.append(torch.stack([layer[sample_index] for layer in all_layer_attentions]))

        return all_sample_layer_attentions


#pearson相似度计算
def pearson_correlation(x, y,e):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    numerator = torch.sum(x_centered * y_centered)

    # 计算分母：标准差的乘积
    denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2) + e)
    r = numerator / denominator
    r = 1 - r
    return r


#自定义两个分布之间的wasserstein_distance
def custom_wasserstein_distance(s_atten,t_atten):
    x = torch.arange(s_atten.shape[-1])
    y = torch.arange(t_atten.shape[-1])
    w_x = s_atten
    w_y = t_atten

    # 1. 对分布进行排序
    _, idx_x = torch.sort(x)
    _, idx_y = torch.sort(y)
    
    # 2. 根据排序重排概率质量
    w_x_sorted = w_x[idx_x]
    w_y_sorted = w_y[idx_y]
    
    # 3. 计算 Wasserstein 距离：计算累积的运输代价
    cdf_x = torch.cumsum(w_x_sorted, dim=0)  # 累计分布函数 CDF
    cdf_y = torch.cumsum(w_y_sorted, dim=0)
    
    # 4. 计算 Wasserstein 距离
    distance = torch.sum(torch.abs(cdf_x - cdf_y))
    return distance





