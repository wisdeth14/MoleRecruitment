# MoleRecruitment

Poisoning attack implemented against two different CL repositiories:

LwF, LwM, iCaRL, IL2M: https://github.com/mmasana/FACIL

SSRE, DER: https://github.com/G-U-N/PyCIL



To obtain moles, provide "cifar100" or "imagenet_subset" to the following command:
```
python mole_search.py --datasets "dataset"
```

Use the resulting .npy files to recruit moles to implement the poisonous attack. Follow instructions in the FACIL/PyCIL directories, and append the folllowing to the python command (adjust rho threshold as desired):

```
--moles --rho 0.5
```



