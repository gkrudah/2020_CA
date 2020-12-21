# 2020 Computer Architecture Term Project

In checkpoint directory, each pth file is the best performed model for each type.

- pretrain
  > $ python3 main.py --mode train --type 1

- for specific dir for result
  > $ python3 main.py --mode train --type 1 –checkpoint your_dir

- pseudo train ( either type 2 or 3 )
  > $ python3 main.py --mode train --type 2 --saved_model your_pretrained_model.pth
 
- evaluate saved model
  > $ python3 main.py --mode eval --saved model model_name.pth
