NAME="seqLSTM_vit32"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
main_task_retrieval.py --do_train --num_thread_reader=8 \
--name ${NAME} \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ./datasets/msrvtt_data/MSRVTT_train.9k.csv \
--val_csv ./datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv \
--data_path ./datasets/msrvtt_data/MSRVTT_data.json \
--features_path /home/xinzijie/VisualSearch/msrvtt10k/ImageData \
--output_dir logs \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqLSTM \
--pretrained_clip_name ViT-B/32