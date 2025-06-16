# # Scenario 1: one source and a driver video (as frames)
# python inference.py --source_root ./assets/source_images/adilbek.png \
#                     --driver_root ./assets/driver_video \
#                     --model_config_path ./configs/voodooxp.yml \
#                     --render_mode driver_view \
#                     --batch_size 1 \
#                     --weight_path ./experiments/pretrained_weights/voodooxp_weight_v1.0.ckpt \
#                     --save_root ./results/adilbek_driven_by_phong \
# 
# # Scenario 2: pairwise inference between n sources and m drivers. There are n*m output images
# python inference.py --source_root ./assets/source_images \
#                     --driver_root ./assets/driver_images \
#                     --model_config_path ./configs/voodooxp.yml \
#                     --render_mode driver_view \
#                     --batch_size 1 \
#                     --weight_path ./experiments/pretrained_weights/voodooxp_weight_v1.0.ckpt \
#                     --save_root ./results/pairwise_results \
#                     --pairwise

# Scenario 3: pairs of n sources and n drivers. There are n output images
python inference.py --source_root ./assets/source_images \
                    --driver_root ./assets/driver_images \
                    --model_config_path ./configs/voodooxp.yml \
                    --render_mode driver_view \
                    --batch_size 1 \
                    --weight_path ./experiments/pretrained_weights/voodooxp_weight_v1.0.ckpt \
                    --save_root ./results/n_sources_n_drivers \

# # Scenario 4: one source and a driver video, but you want to use a flying trajectory to render the rennacted face
# python inference.py --source_root ./assets/source_images/adilbek.png \
#                     --driver_root ./assets/driver_video \
#                     --model_config_path ./configs/voodooxp.yml \
#                     --render_mode novel_view \
#                     --batch_size 1 \
#                     --weight_path ./experiments/pretrained_weights/voodooxp_weight_v1.0.ckpt \
#                     --save_root ./results/adilbek_driven_by_phong \
