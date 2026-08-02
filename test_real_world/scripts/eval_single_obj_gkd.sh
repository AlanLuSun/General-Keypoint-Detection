#!/bin/bash
#SBATCH --job-name=eval          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus-per-node=1        # number of GPUs per node(only valid under large/normal partition)
#SBATCH --cpus-per-task=28       # number of CPUs (28, 56, 112, 224 for 1, 2, 4, 8 GPUs)
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=vonneumann   # partition(large/normal/cpu) where you submit
#SBATCH --account=vonneumann1    # only require for multiple projects

# common paths: /home/changshenglu, /project/vonneumann1/changsheng, /scratch/vonneumann1/changsheng
# interactive job: srun --partition vonneumann --account=vonneumann1 --nodes=1 --gpus-per-node=1 --cpus-per-task=28 --pty bash
# login to a running job: srun --jobid=502542 -w dgx-50 --overlap --pty bash -i

# we install environments ...

module purge  # clear environment modules inherited from submission
module load slurm cuda11.8/blas/11.8.0 cuda11.8/fft/11.8.0 cuda11.8/toolkit/11.8.0  # need Pytorch with at least cuda11.8 
source /home/changshenglu/anaconda3/bin/activate openkd
#source activate /project/vonneumann1/cl2025/anaconda_envs/openkd

cd '/project/vonneumann1/cl2025/GKD'
# cd '/home/changshenglu/GKD'
echo $(pwd)

python3 -m debugpy --listen 10.22.4.152:4567 --wait-for-client \
    test_real_world/single_obj_gkd_inference.py \
    --cfg_file /project/vonneumann1/cl2025/GKD_github/test_real_world/configs/gkd.yaml \
    --checkpoint /project/vonneumann1/cl2025/GKD/output/gkd_ablation/main5_fullset_run/fullset_kg_blk2/model/gkd_fullset.best \
    --input_im /project/vonneumann1/cl2025/GKD_github/test_real_world/ims1/2007_007524.jpg \
    --bbox_on_input_im \
    --obj_type '' \
    --kps_texts 'left eye' 'right eye' 'nose' \
    --support_im /project/vonneumann1/cl2025/GKD_github/test_real_world/ims1/alpaca_150.jpg \
    --support_kps 615 495 483 493 521 549 \
    --skeleton 1 2 1 3 2 3
    # MODEL.ENCODER.DINOv3.VISUAL_ENCODER dinov3_vith16plus 


python3 -m debugpy --listen 10.22.4.152:4567 --wait-for-client \
    test_real_world/single_obj_gkd_inference.py \
    --cfg_file /project/vonneumann1/cl2025/GKD/test_real_world/configs/gkd.yaml \
    --checkpoint /project/vonneumann1/cl2025/GKD/output/gkd_ablation/main5_fullset_run/fullset_kg_blk2/model/gkd_fullset.best \
    --input_im /project/vonneumann1/cl2025/GKD/test_real_world/ims1/2007_007524.jpg \
    --bbox_on_input_im 33 38 241 310 \
    --obj_type '' \
    --kps_texts 'nose' 'left eye' 'right eye' 'left ear' 'right ear' \
    --skeleton 1 2 1 3 2 3 2 4 3 5


python3 -m debugpy --listen 10.22.4.90:4567 --wait-for-client \
    test_real_world/single_obj_gkd_inference.py \
    --cfg_file /project/vonneumann1/cl2025/GKD/test_real_world/configs/gkd.yaml \
    --checkpoint /project/vonneumann1/cl2025/GKD/output/gkd_ablation/main5_fullset_run/fullset_kg_blk2/model/gkd_fullset.best \
    --input_im /project/vonneumann1/cl2025/GKD/test_real_world/ims1/2007_007524.jpg \
    --bbox_on_input_im \
    --obj_type "" \
    --kps_texts 'left eye' 'right eye' 'nose' \
    --support_im /project/vonneumann1/cl2025/GKD/test_real_world/ims1/2007_003778.jpg \
    --support_kps 343 166 281 158 311 197 \
    --skeleton 1 2 1 3 2 3
