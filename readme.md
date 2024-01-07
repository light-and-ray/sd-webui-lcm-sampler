# LCM Sampler

It is an extention, wich packs diff from this issue in an extention:
https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13952

![](images/img1.jpg)

Use this sampler to not break generation, using lcm-loras: 
- https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
- https://huggingface.co/latent-consistency/lcm-lora-ssd-1b
- https://huggingface.co/latent-consistency/lcm-lora-sdxl

Or you can use it with [sd_turbo](models/Stable-diffusion/sd_turbo.safetensors). It is turbo version of sd 2.1 512. Useful with amazing [StableSR](https://github.com/pkuliyi2015/sd-webui-stablesr) uspcaling


There is no diffirence in taken time between lcm and euler a samplers

![](images/img3.jpg)

Plot for steps 1-20:
![](images/img2.jpg)
