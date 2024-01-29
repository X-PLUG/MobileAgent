# Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception
<div align="center">
    <a href="https://arxiv.org/abs/2311.07397"><img src="assets/Paper-Arxiv-orange.svg" ></a>
</div>
<br>
<div align="center">
Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan,
</div>
<div align="center">
Weizhou Shen, Ji Zhang, Fei Huang, Jitao Sang
</div>
<br>
<br>

![](assets/example.jpg?v=1&type=image)

## News

* [1.30] 🔥Our paper is available at [LINK]().
* [1.30] 🔥Our evaluation results on Mobile-Eval are available at [LINK]().
* [1.30] The code and Mobile-Eval benchmark are coming soon!

## Demo
https://github.com/X-PLUG/MobileAgent/assets/127390760/26c48fb0-67ed-4df6-97b2-aa0c18386d31

## Mobile-Eval
Mobile-Eval is a benchmark designed for evaluating the performance of mobile device agents. This benchmark includes 10 mainstream single-app scenarios and 1 multi-app scenario. 

For each scenario, we have designed three instructions:
* Instruction 1: relatively simple and basic task
* Instruction 2: additional requirements added on top of the difficulty of Instruction 1
* Instruction 3: user demands with no explicit task indication


## Evaluation results
We evaluated Mobile-Agent on Mobile-Eval. The evaluation results are available at [LINK].
*   We have stored the evaluation results for the 10 apps and the multi-app scenario in folders named after each app.
* The numbers within each app's folder represent the results for different types of instruction within that app.
*   For example, if you want to view the results of Mobile-Agent for the second instruction in Google Maps, you should go to the following path:```results/Google Maps/2```.
* If the last action of Mobile-Agent is not "stop", it indicates that Mobile-Agent did not complete the corresponding instruction. During the evaluation, we manually terminated these cases where completion was not possible.

## Citation
If you found this work useful, consider giving this repository a star and citing our paper as followed:
```

```

