import os
import shutil
import torch
from transformers import Seq2SeqTrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainerCallback
from transformers import T5ForConditionalGeneration
from transformers import  GPT2LMHeadModel
from mixture_layer import StudentModel
from transformers.trainer_utils import set_seed
import logging
import time
from tqdm import tqdm
from model_utils import TaskPrefixDataCollator, Decoder_TaskPrefixTrainer, StandardDataCollator, Decoder_StandardTrainer

#自定义日志
class CustomEvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", {})
        print(f"Evaluation metrics in callback: {metrics}")


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, "log.jsonl")
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w', encoding='utf-8') as f:
            pass

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",  # 去掉日志等级
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode='w'
    )


def get_config_dir(args):
    #根据数据集、学生模型、方法选择、教师模型选择、是否在训练集上进行了子采样以及各个超参数创建文件夹保存模型
    if args.attention_distill:
        return f'{args.dataset}/{args.from_pretrained.split("/")[-1]}/{args.llm}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}/{args.run}/attention_distill/kl_loss_only/max_steps{args.max_steps}/eval_steps{args.eval_steps}/beta{args.beta}__T_atten_layer{args.t_atten_layer}__S_atten_layer{args.s_atten_layer}__kl_temperature{args.kl_temperature}'
    elif args.MOL_attention_distill:
        return f'{args.dataset}/{args.from_pretrained.split("/")[-1]}/{args.llm}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}/{args.run}/MOL_attention_distill/kl_loss_only/max_steps{args.max_steps}/eval_steps{args.eval_steps}/beta{args.beta}__kl_temperature{args.kl_temperature}__mol_temperature{args.mol_temperature}__t_temperature{args.t_temperature}'
    elif args.MMI_distill:
        return f'{args.dataset}/{args.from_pretrained.split("/")[-1]}/{args.llm}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}/{args.run}/MMI_baseline/max_steps{args.max_steps}/eval_steps{args.eval_steps}/beta{args.beta}' 
    elif args.task_type == 'standard':
        return f'{args.dataset}/{args.from_pretrained.split("/")[-1]}/{args.llm}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}/{args.run}/standard/max_steps{args.max_steps}/eval_steps{args.eval_steps}'
    else:
        return f'{args.dataset}/{args.from_pretrained.split("/")[-1]}/{args.llm}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}/{args.run}/baseline/max_steps{args.max_steps}/eval_steps{args.eval_steps}'

def get_layer_weight_path(args, log_dir):
    if  args.MOL_attention_distill:
        return f'{log_dir}/s_t_layer_weight.jsonl'
    else:
        return None
    
def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics):
    set_seed(run)
    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}' 
    # logging_dir = f'logs/{config_dir}'
    # setup_logging(logging_dir)
    # logging.info(args)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        #加载不同的学生模型
        if args.MOL_attention_distill:
            model = StudentModel(student_model_path=args.from_pretrained, temperature=args.mol_temperature)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.from_pretrained, torch_dtype=torch.bfloat16)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.MOL_attention_distill:
            model = StudentModel(student_model_path=args.from_pretrained, temperature=args.mol_temperature).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.from_pretrained, torch_dtype=torch.bfloat16).to(device)

    layer_weight_path = None


    #Seq2SeqTrainingArguments 对象，包含训练的详细配置，如学习率、批次大小、训练步数等
    training_args = Seq2SeqTrainingArguments(
        output_dir, #用于保存模型检查点和其他训练输出的目录
        deepspeed=args.deepspeed,
        remove_unused_columns = False, #当设置为 False 时，不会删除数据集中未使用的列。对于多任务训练，通常需要保留未使用的列（例如解释标签等）用于自定义数据整理
        evaluation_strategy = 'steps', #定义模型评估的频率， 'steps'：每隔一定训练步数进行评估
        eval_steps=args.eval_steps, #控制评估的步数间隔，每训练 eval_steps 步进行一次评估
        save_strategy='steps', #定义模型保存的频率，'no'：不进行模型保存,'steps'：按步数保存
        save_steps=args.max_steps, #模型保存的步数间隔。与 save_strategy 配合使用，若 save_strategy='steps'，则每隔 save_steps 步保存一次
        # metric_for_best_model="eval_test_accuracy",  # 使用 eval_test_accuracy 来选择最佳模型
        greater_is_better=True,             # 准确率越高越好
        save_total_limit=1,                 # 只保留最新的1个检查点
        logging_dir=None, #日志文件的保存目录，用于 TensorBoard 或其他日志追踪工具
        logging_strategy="no", #控制日志记录的频率
        logging_steps=1, #日志记录的步数间隔。与 logging_strategy 配合使用，若 logging_strategy='steps'，则每隔 logging_steps 步记录一次日志
        max_steps=args.max_steps, #训练的总步数。在达到此步数后，训练停止。它覆盖 num_train_epochs 的值，直接设置训练结束的总步数
        learning_rate=args.lr, 
        gradient_accumulation_steps=args.grad_steps, #梯度累积步数。通过累积若干步的梯度再进行反向传播，可以在小显存的设备上使用较大批次大小的效果
        per_device_train_batch_size=args.batch_size, #每个设备（GPU/TPU）上的训练批次大小。实际的批次大小为 per_device_train_batch_size * num_devices
        per_device_eval_batch_size=args.test_batch_size, #每个设备上的评估批次大小，设置方式与 per_device_train_batch_size 类似
        predict_with_generate=True, #当设置为 True 时，使用 generate() 方法进行预测（通常用于生成任务，如机器翻译、文本摘要），而不是直接输出模型的 logits
        seed=run, #设置随机种子
        local_rank=local_rank, #用于分布式训练时指定每个设备的编号，通常自动分配。默认值为 -1，表示不使用分布式训练
        bf16=args.bf16, #设置是否启用 bfloat16 精度训练。如果设备支持 bfloat16，可以减少显存占用并加速计算
        fp16=False,  # 如果使用bf16则禁用fp16
        generation_max_length=args.gen_max_len, #生成序列时的最大长度，通常用于控制生成任务输出的长度
        prediction_loss_only=False, #设置为 True，则只返回预测损失，而不返回其他指标或预测结果。用于降低内存占用或在特定场景下简化输出
    )

    if args.task_type == 'task_prefix':
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    elif args.task_type == 'standard':
        data_collator = StandardDataCollator(tokenizer=tokenizer, model=model)
    else:
        raise ValueError

    #自定义日志
    custom_eval_callback = CustomEvalCallback()

    trainer_kwargs = {
        'alpha': args.alpha,
        'beta':args.beta,
        'attention_distill':args.attention_distill,
        's_atten_layer': args.s_atten_layer,
        'kl_temperature': args.kl_temperature,
        'MOL_attention_distill': args.MOL_attention_distill,
        "layer_weight_path": layer_weight_path,
        'MMI_distill':args.MMI_distill,
        't_temperature' : args.t_temperature,
        'output_rationale': args.output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': {'test': tokenized_datasets["test"],},
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
        'callbacks': [custom_eval_callback] 
    }
    
    
    if args.task_type == 'task_prefix':
            trainer = Decoder_TaskPrefixTrainer(**trainer_kwargs)
            print("*********************************decoder trainer************************************")
    
    elif args.task_type == 'standard':
        trainer_kwargs.pop('alpha')
        trainer_kwargs.pop('output_rationale')
        trainer_kwargs.pop('attention_distill')
        trainer_kwargs.pop('beta')
        trainer_kwargs.pop('MMI_distill')
        trainer_kwargs.pop('t_temperature')
        trainer_kwargs.pop('s_atten_layer')
        trainer_kwargs.pop('MOL_attention_distill')
        trainer_kwargs.pop('layer_weight_path')
        trainer_kwargs.pop('kl_temperature')

        trainer = Decoder_StandardTrainer(**trainer_kwargs)
    else:
        raise ValueError
    
    if args.ood_evalu_only:
        trainer.evaluate()
    else:
        trainer.train()
    
  