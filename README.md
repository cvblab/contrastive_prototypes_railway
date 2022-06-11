## Supervised contrastive learning-guided prototypes for railway crossing inspections

This repository contains implementation code for the paper 'Supervised contrastive learning-guided prototypes on axle-box accelerations for railway crossing inspections'. In particular, this work tackels the challenge of using deep learning models on scenarios with limited data. In different industrial scenarios, collection large datasets is a challenging tasks, since data itseld is available or costly. Computer vision algorithms based con deep learning are data-hungry, and usually require huge amounts of data to perform properly. In this work, the use of supervised contrastive learning as a proxy for prototypical-based inference is proposed to alleviate this issue, in a few-shot learning fashion.

![ssl](https://github.com/cvblab/contrastive_prototypes_railway/blob/main/repo_images/method.png)

You can train the proposed method using:

```
python main_prototypical.py
```

If you consider this work usefull for your reseach, please consider citing:

**J. Silva-Rodríguez, P. Salvador, V. Naranjo and R. Insa, "Supervised contrastive learning-guided prototypes on axle-box accelerations for railway crossing inspections". Preprint (2022).**
