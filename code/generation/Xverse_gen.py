import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from model_gen import model_generator

class XVERSE_generator(model_generator):
    def __init__(self, f_path, is_few_shot, device, is_vllm, model_path, model_name, few_shot_path, tensor_parallel_size, gpu_memory_utilization):
        '''
        Args:
            model_name: The name of the model
        '''
        super(XVERSE_generator, self).__init__(f_path, is_few_shot, device, is_vllm, model_path, few_shot_path, tensor_parallel_size, gpu_memory_utilization)
        self.model_name = model_name
        
    def model_init(self):
        '''
        Init the model
        '''
        if self.is_vllm:
            model, tokenizer = self.model_init_vllm()
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16).eval()
        return model, tokenizer
                
    def generate_output(self, q_type, tokenizer, model, batch_size=None):
        '''
        Generate the output for XVERSE_13B, using the suitable prompt template
        '''
        task_name = self.f_path.split("/")[-1].split(".")[0]
        instruction_ls, input_text_ls, answer_ls = self.process_prompt(task_name)
        output_ls = []
        num_unsuccess = 0
        if self.is_vllm:
            sampling_params = SamplingParams(temperature=0, max_tokens=20) if q_type == 'multiple_choice' else SamplingParams(temperature=0, max_tokens=512)
            for idx in range(0, len(instruction_ls), batch_size):
                instruction, input_text, answer = instruction_ls[idx:idx+batch_size], input_text_ls[idx:idx+batch_size], answer_ls[idx:idx+batch_size]
                prompt = [instruction[i] + input_text[i] for i in range(len(instruction))]
                # Truncate the prompt
                prompt = [model_generator.truncate_long(prompt[i], 4096, tokenizer, q_type) for i in range(len(prompt))]
                try:
                    response_ls = model.generate(prompt, sampling_params)
                    for idx, out in enumerate(response_ls):
                        response = out.outputs[0].text
                        output_ls.append({"input": input_text[idx],
                                    "output": response,
                                    "answer": answer[idx]})
                except:
                    for i in range(len(instruction)):
                        num_unsuccess += 1
                        print("Fail to answer")
                        response = "未成功回答"
                        output_ls.append({"input": input_text,
                                  "output": response,
                                  "answer": answer})
        else:
            for instruction, input_text, answer in tqdm(zip(instruction_ls, input_text_ls, answer_ls), total=len(input_text_ls)):
                prompt = instruction + input_text
                # Truncate the prompt
                prompt = model_generator.truncate_long(prompt, 4096, tokenizer, q_type)
                inputs = tokenizer(prompt, return_tensors="pt").input_ids
                inputs = inputs.cuda()
                try:
                    generated_ids = model.generate(input_ids=inputs, max_new_tokens=20, do_sample=False) if q_type == "multiple_choice" else model.generate(input_ids=inputs, max_new_tokens=512, do_sample=False)
                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    response = "".join(response)
                    response = response.split("答案:")[-1]
                except:
                    num_unsuccess += 1
                    print("Fail to answer")
                    response = "未成功回答"
                output_ls.append({"input": input_text,
                                "output": response,
                                "answer": answer})
        return output_ls, num_unsuccess