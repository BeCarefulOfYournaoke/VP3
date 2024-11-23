# VP3
Rethinking Multi-pattern Mining from a Perspective of Pattern Prototype Learning
---

## Notice-1

The configuration of VP3 for 4 datasets, CIFAR100, Travel, ILSVRC20, Place20 is in folder `config`.

training by running:

```python
python main.py -c ./config/multiCom_Cls_Con_Cifar.yaml  
python main.py -c ./config/multiCom_Cls_Con_Travel.yaml  
python main.py -c ./config/multiCom_Cls_Con_Place.yaml 
python main.py -c ./config/multiCom_Cls_Con_ILSVRC.yaml
```

visualization by running:

```python
# edit the 'modle_save_path' and 'weight_save_path' in xxx.yaml
python Visual.py -c ./config/multiCom_Cls_Con_Cifar.yaml
```


## Notece-2

The CIFAR100 datasets has two hierarchical structure with `CIFAR20/class-1/beaver`, `CIFAR20/class-1/dolphin`, `CIFAR20/class-1/otter`, `CIFAR20/class-1/seal`,`CIFAR20/class-1/whale`

The other datasets has common structure with `dataset/class-1`, `dataset/class-2`, ……



