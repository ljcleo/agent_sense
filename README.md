# AgentSense: Benchmarking Social Intelligence of Language Agents through Interactive Scenarios

This is the code repository for *AgentSense: Benchmarking Social Intelligence of Language Agents through Interactive Scenarios*.

- [Paper](https://arxiv.org/abs/2410.19346)


## Model Performance
| Models        | Goal      |           |              |                   |                  |               |                |          | Info      |           |
| ------------- | --------- | --------- | ------------ | ----------------- | ---------------- | ------------- | -------------- | -------- | --------- | --------- |
|               | self      | other     | judge-GPT-4o | judge-Qwen2.5-72B | judge-Llama3-70B | judge-Average | judge-Majority | PSI      | Acc.      | PSI       |
| Llama-2-7B    | 83.38     | 62.70     | 52.73        | 57.68             | 55.37            | 55.26         | 55.84          | 21.94    | 33.06     | 20.53     |
| Llama-2-13B   | 48.01     | 10.26     | 17.38        | 30.11             | 72.19            | 39.90         | 30.91          | 21.84    | 28.56     | 18.39     |
| Llama-2-70B   | 85.72     | 65.65     | 33.78        | 42.37             | 73.80            | 49.98         | 45.53          | 22.31    | 36.78     | 18.60     |
| Llama-3-8B    | 87.63     | 67.28     | 79.90        | 82.55             | 75.10            | 79.18         | 80.71          | 12.85    | 69.68     | 15.14     |
| Llama-3-70B   | 80.38     | 77.27     | 86.22        | 87.61             | 79.88            | 84.57         | 86.27          | 8.92     | 73.08     | 16.58     |
| Qwen2.5-7B    | 86.17     | 61.92     | 77.07        | 79.30             | 71.99            | 76.12         | 77.37          | 13.10    | 74.82     | 15.84     |
| Qwen2.5-14B   | 86.62     | 84.17     | 88.43        | **89.83**         | 80.47            | 86.24         | 88.14          | 8.09     | 75.02     | 14.81     |
| Qwe2.5-72B    | 90.67     | 85.89     | 88.29        | 89.03             | 78.57            | 85.30         | 87.74          | 8.19     | 76.05     | **13.57** |
| Mistral-7B    | **95.22** | **87.25** | 79.29        | 84.13             | 77.82            | 80.41         | 82.37          | 12.39    | 66.59     | 18.55     |
| GPT-3.5-turbo | 90.16     | 76.62     | 82.12        | 84.37             | 77.30            | 81.26         | 82.64          | 10.01    | 68.41     | 18.37     |
| GPT-4o        | 88.46     | 86.29     | **88.47**    | 89.00             | **81.57**        | **86.34**     | **88.36**      | **6.99** | **76.86** | 15.48     |



## Simulation and Evaluation
For simulation and evalution, please refer to [README](https://github.com/ljcleo/agent_sense/tree/main/SENSE).


## Citation

If you find our AgentSense benchmark useful, please consider citing our paper:

```BibTeX
@Article{Mou2024AgentSense,
  title={AgentSense: Benchmarking Social Intelligence of Language Agents through Interactive Scenarios},
  author={Mou, Xinyi and Liang, Jingcong and Lin, Jiayu and Zhang, Xinnong and Liu, Xiawei and Yang, Shiyue and Ye, Rong and Chen, Lei and Kuang, Haoyu and Huang, Xuanjing and Wei, Zhongyu},
  journal={arXiv preprint arXiv:2410.19346},
  year={2024}
}

```
