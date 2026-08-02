# GKD's bundled DINOv3 implementation requires Python >= 3.10.  Do not use the
# GroundingDINO conda environment here: it uses Python 3.9.
PYTHON_BIN=/project/vonneumann1/cl2025/anaconda_envs/openkd/bin/python

# "${PYTHON_BIN}" -m test_real_world.multi_obj_gkd_inference \
#     --cfg_file /project/vonneumann1/cl2025/GKD/test_real_world/configs/gkd.yaml \
#     --checkpoint /project/vonneumann1/cl2025/GKD/output/gkd_ablation/main5_fullset_run/fullset_kg_blk2/model/gkd_fullset.best \
#     --input_im /project/vonneumann1/cl2025/GKD/test_real_world/ims1/alpaca_150.jpg \
#     --obj_type 'alpaca' \
#     --kps_texts 'left eye' 'right eye' 'nose' \
#     --support_im /project/vonneumann1/cl2025/GKD/test_real_world/ims1/alpaca_150.jpg \
#     --support_kps 615 495 483 493 521 549 \
#     --skeleton 1 2 1 3 2 3  \
#     --object_detector locateanything

# "${PYTHON_BIN}" -m test_real_world.multi_obj_gkd_inference \
#     --cfg_file /project/vonneumann1/cl2025/GKD/test_real_world/configs/gkd.yaml \
#     --checkpoint /project/vonneumann1/cl2025/GKD/output/gkd_ablation/main5_fullset_run/fullset_kg_blk2/model/gkd_fullset.best \
#     --input_im /project/vonneumann1/cl2025/GKD/test_real_world/ims1/alpaca_150.jpg \
#     --obj_type 'alpaca' \
#     --kps_texts 'left eye' 'right eye' 'nose'

# "${PYTHON_BIN}" -m test_real_world.multi_obj_gkd_inference \
#     --cfg_file /project/vonneumann1/cl2025/GKD/test_real_world/configs/gkd.yaml \
#     --checkpoint /project/vonneumann1/cl2025/GKD/output/gkd_ablation/main5_fullset_run/fullset_kg_blk2/model/gkd_fullset.best \
#     --input_im /project/vonneumann1/cl2025/GKD/test_real_world/ims1/2007_005460.jpg \
#     --obj_type 'cat' \
#     --object_detector locateanything

# "${PYTHON_BIN}" -m test_real_world.multi_obj_gkd_inference \
#     --cfg_file /project/vonneumann1/cl2025/GKD/test_real_world/configs/gkd.yaml \
#     --checkpoint /project/vonneumann1/cl2025/GKD/output/gkd_ablation/main5_fullset_run/fullset_kg_blk2/model/gkd_fullset.best \
#     --input_im /project/vonneumann1/cl2025/GKD/test_real_world/ims1/cat_dog.jpg \
#     --obj_type 'cat' \
#     --object_detector locateanything

"${PYTHON_BIN}" -m test_real_world.multi_obj_gkd_inference \
    --cfg_file /project/vonneumann1/cl2025/GKD/test_real_world/configs/gkd.yaml \
    --checkpoint /project/vonneumann1/cl2025/GKD/output/gkd_ablation/main5_fullset_run/fullset_kg_blk2/model/gkd_fullset.best \
    --input_im /project/vonneumann1/cl2025/GKD/test_real_world/ims1/alpaca_150.jpg \
    --obj_type 'cat, alpaca' \
    --object_detector locateanything


#========================================================================
# Notes to reproduce environment
#========================================================================
conda create --name env_test python=3.10.4
# Ensure we have one GPU (mount GPU and load modules related to CUDA toolkit in SuperPOD)
conda activate env_test
pip3 install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip3 install opencv-contrib-python
pip3 install -r requirements.txt
# If happens ModuleNotFoundError: No module named 'pkg_resources', lower setuptools version
pip3 install setuptools==78.1.1


# 1. Go to test_real_world/object_detor_lib/weights and download object detection models 
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
git lfs install
git clone https://huggingface.co/nvidia/LocateAnything-3B
# 2. Activate environments and then run python file
source activate /project/vonneumann1/cl2025/anaconda_envs/openkd
# 3. Run the python file
python3 test_real_world/multi_obj_gkd_inference.py \
    --cfg_file /project/vonneumann1/cl2025/GKD_github/test_real_world/configs/gkd.yaml \
    --checkpoint /project/vonneumann1/cl2025/GKD/output/gkd_ablation/main5_fullset_run/fullset_kg_blk2/model/gkd_fullset.best \
    --input_im /project/vonneumann1/cl2025/GKD_github/test_real_world/ims1/alpaca_150.jpg \
    --obj_type 'cat, alpaca' \
    --object_detector locateanything

python3 test_real_world/multi_obj_gkd_inference.py \
    --cfg_file /project/vonneumann1/cl2025/GKD_github/test_real_world/configs/gkd.yaml \
    --checkpoint /project/vonneumann1/cl2025/GKD/output/gkd_ablation/main5_fullset_run/fullset_kg_blk2/model/gkd_fullset.best \
    --input_im /project/vonneumann1/cl2025/GKD_github/test_real_world/ims1/alpaca_150.jpg \
    --obj_type 'cat, alpaca' \
    --object_detector groundingdino