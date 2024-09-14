NAME="tightTransfq_vit32"
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 \
main_task_retrieval.py --do_train --num_thread_reader=8 \
--name ${NAME} \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path /home/xinzijie/Projects/CLIP4Clip-annotated/datasets/msvd_data_origin \
--features_path /home/xinzijie/VisualSearch/msvd/ImageData \
--output_dir logs \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msvd \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header tightTransf \
--pretrained_clip_name ViT-B/32