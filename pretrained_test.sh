# Test bash file for the pre-trained model,  
# you can test on the specified data set using the pre-trained model weights in the ./pretrained_models.
# The pre-training weights are the weights of the uC3DU-Net model.

# IIIIIIIIIII   M       M      OOOOOOOOO      PPPPPPPPP   
#      I        MM     MM     O         O     P       P  
#      I        M M   M M    O           O    P       P  
#      I        M  M M  M   O             O   PPPPPPPPP  
#      I        M   M   M    O           O    P          
#      I        M       M     O         O     P          
# IIIIIIIIIII   M       M      OOOOOOOOO      P          

# Create on 2024-6-1 Saturday.
# @author: jjhuang and tyler


### Initial hyperparameter
ROOT="./datasets/OIMHS" # Relative path to the dataset
OUTPUT="./pretrained_models/OIMHS"  # the path to the pretrained model folder
DATASET="OIMHS"  # dataloader
NETWORKS="uC_3DUNet" # You need to select the network you want to train here, please refer to ./model.py for details.
NUM_CLASS='5'  # You need to replace num_class with the number of segmentation categories for the specified dataset, including background.

########################  test ##########################
torchrun --nproc_per_node=1 --master_port=29500 ./test.py \
    --root $ROOT \
    --output $OUTPUT/test \
    --dataset $DATASET \
    --network  $NETWORKS\
    --trained_weights $OUTPUT/OIMHS.pth \
    --mode test \
    --in_channel 1 \
    --out_classes $NUM_CLASS \
    --sw_batch_size 2 \
    --overlap 0.5 \
    --cache_rate 0 \
    --world_size 1 \
    --testrank 0 \
    --distributed 

### You can adjust the metrics calculated by modifying 'metrics_list', which detailed in./utils/niigz2excel.py
python ./utils/niigz2excel.py \
    --root $ROOT \
    --output $OUTPUT/final \
    --network $NETWORKS \
    --metrics_list iou dice assd hd hd95 \
    --out_classes $NUM_CLASS

###  Paste the following command into the command line to quickly start the testing process, 
###  you need to replace 3dseg with your virtual environment.

: <<'END'

kill -9 $(lsof -t -i:29500)
lsof -i :29500

source activate 3dseg
bash pretrained_test.sh

END