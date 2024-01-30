# Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception
<div align="center">
    <a href="https://arxiv.org/abs/2311.07397"><img src="assets/Paper-Arxiv-orange.svg" ></a>
</div>
<br>
<div align="center">
Junyang Wang<sup>1</sup>, Haiyang Xu<sup>2†</sup>, Jiabo Ye<sup>2</sup>, Ming Yan<sup>2†</sup>,
</div>
<div align="center">
Weizhou Shen<sup>2</sup>, Ji Zhang<sup>2</sup>, Fei Huang<sup>2</sup>, Jitao Sang<sup>1†</sup>
</div>
<div align="center">
{junyangwang, jtsang}@bjtu.edu.cn, {shuofeng.xhy, ym119608}@alibaba-inc.com
</div>
<br>
<div align="center">
<sup>1</sup>Beijing Jiaotong University    <sup>2</sup>Alibaba Group
</div>
<div align="center">
† Corresponding author
</div>
<br>
<br>

![](assets/example.jpg?v=1&type=image)

## News

* [1.30] 🔥Our paper is available at [LINK](https://arxiv.org/abs/2401.16158).
* [1.30] 🔥Our evaluation results on Mobile-Eval are available.
* [1.30] The code and Mobile-Eval benchmark are coming soon!

## Demo
https://github.com/X-PLUG/MobileAgent/assets/127390760/26c48fb0-67ed-4df6-97b2-aa0c18386d31

## Mobile-Eval
Mobile-Eval is a benchmark designed for evaluating the performance of mobile device agents. This benchmark includes 10 mainstream single-app scenarios and 1 multi-app scenario. 

For each scenario, we have designed three instructions:
* Instruction 1: relatively simple and basic task
* Instruction 2: additional requirements added on top of the difficulty of Instruction 1
* Instruction 3: user demands with no explicit task indication

The detailed content of Mobile-Eval is as follows:
| Application   | Instruction   |
|-------|-------|
| Alibaba.com   | 1. Help me find caps in Alibaba.com.<br>2. Help me find caps in Alibaba.com. If the "Add to cart" is avaliable in the item information page, please add the item to my cart.<br>3. I want to buy a cap. I've heard things are cheap on Alibaba.com. Maybe you can find it for me. |
| Amazon Music   | 1. Search singer Jay Chou in Amazon Music.<br>2. Search a music about "agent" in Amazon Music and play it.<br>3. I want to listen music to relax. Find an App to help me. |
| Chrome   | 1. Search result for today's Lakers game.<br>2. Search the information about Taylor Swift.<br>3. I want to know the result for today's Lakers game. Find an App to help me. |
| Gmail   | 1. Send an empty email to to {address}.<br>2. Send an email to {address}n to tell my new work.<br>3. I want to let my friend know my new work, and his address is {address}. Find an App to help me. |
| Google Maps   | 1. Navigate to Hangzhou West Lake.<br>2. Navigate to a nearby gas station.<br>3. I want to go to Hangzhou West Lake, but I don't know the way. Find an App to help me. |
| Google Play   | 1. Download WhatsApp in Play Store.<br>2. Download Instagram in Play Store.<br>3. I want WhatsApp on my phone. Find an App to help me. |
| Notes   | 1. Create a new note in Notes.<br>2. Create a new note in Notes and write "Hello, this is a note", then save it.<br>3. I suddenly have something to record, so help me find an App and write down the following content: meeting at 3pm. |
| Settings   | 1. Turn on the dark mode.<br>2. Turn on the airplane mode.<br>3. I want to see the real time internet speed at the battery level, please turn on this setting for me. |
| TikTok   | 1. Swipe a video about pet cat in TikTok and click a "like" for this video.<br>2. Swipe a video about pet cat in TikTok and comment "Ohhhh, so cute cat!".<br>3. Swipe videos in TikTok. Click "like" for 3 pet video cat. |
| YouTube  | 1. Search for videos about Stephen Curry on YouTube.<br>2. Search for videos about Stephen Curry on YouTube and open "Comments" to comment "Oh, chef, your basketball spirit has always inspired me".<br>3. I need you to help me show my love for Stephen Curry on YouTube. |
| Multi-App  | 1. Open the calendar and look at today's date, then go to Notes and create a new note to write "Today is {today's data}".<br>2. Check the temperature in the next 5 days, and then create a new note in Notes and write a temperature analysis.<br>3. Search the result for today's Lakers game, and then create a note in Notes to write a sport news for this result. |

## Evaluation results
We evaluated Mobile-Agent on Mobile-Eval. The evaluation results are available at [LINK](https://github.com/X-PLUG/MobileAgent/tree/main/results).
*   We have stored the evaluation results for the 10 apps and the multi-app scenario in folders named after each app.
* The numbers within each app's folder represent the results for different types of instruction within that app.
*   For example, if you want to view the results of Mobile-Agent for the second instruction in Google Maps, you should go to the following path:```results/Google Maps/2```.
* If the last action of Mobile-Agent is not "stop", it indicates that Mobile-Agent did not complete the corresponding instruction. During the evaluation, we manually terminated these cases where completion was not possible.

## Citation
If you found this work useful, consider giving this repository a star and citing our paper as followed:
```

```

